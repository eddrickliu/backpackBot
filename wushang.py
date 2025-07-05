#!/usr/bin/env python3
"""
Backpack SOL Spot-Futures Arbitrage System (Spot Limit, Futures Market)

Required Backpack Client Interface:
The client object must implement these async methods:
- get_orderbook(symbol) -> Dict with 'bids' and 'asks' arrays
- create_limit_order(symbol, side, price, quantity, post_only=True) -> Dict with 'id'
- create_market_order(symbol, side, quantity) -> Dict with 'id' and 'price'
- get_order(order_id) -> Dict with 'status' and 'executed_quantity'
- cancel_order(order_id) -> Dict
"""

import time
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackpackArbitrageConfig:
    """Configuration for the arbitrage system"""
    def __init__(self):
        # Trading pairs
        self.spot_symbol = "SOL/USDC"
        self.market_symbol = "SOL_USDC"
        self.futures_symbol = "SOL-PERP"
       
        # Risk parameters
        self.max_position_limit = 10000  # USD value
        self.max_order_size = 1000       # USD per order
        self.min_basis_threshold = 0.0005  # 0.05% (Backpack market fee on futures)
       
        # Based on historical data analysis
        self.target_basis_long = 0.001   # 0.1% - conservative threshold for long basis
        self.target_basis_short = -0.001  # -0.1% - conservative threshold for short basis
        self.extreme_basis_close = 0.10   # 10% - close positions when basis is extreme
       
        # Position management
        self.max_funding_periods = 3      # Close after 3 funding periods (24 hours)
        self.emergency_close_loss = 100   # USD loss trigger
       
        # Order execution
        self.limit_order_timeout = 30     # seconds
        self.order_check_interval = 0.1   # seconds
        self.price_update_interval = 2    # seconds - how often to update limit order prices
        self.min_fill_size = 0.001        # Minimum fill size to trigger hedge

class MarketDataAnalyzer:
    """Analyze market data and calculate basis"""
   
    def __init__(self, config: BackpackArbitrageConfig):
        self.config = config
        self.basis_history = []
       
    async def get_orderbook_data(self, public_client, client) -> Dict:
        """Get orderbook data for both spot and futures"""
        try:
            # Get spot orderbook
            spot_ob = public_client.get_depth(self.config.market_symbol)
            spot_best_bid = {
                'price': float(spot_ob['bids'][0][0]),
                'size': float(spot_ob['bids'][0][1])
            }
            spot_best_ask = {
                'price': float(spot_ob['asks'][0][0]),
                'size': float(spot_ob['asks'][0][1])
            }
           
            # Get futures orderbook
            futures_ob = public_client.get_depth(self.config.market_symbol)
            futures_best_bid = {
                'price': float(futures_ob['bids'][0][0]),
                'size': float(futures_ob['bids'][0][1])
            }
            futures_best_ask = {
                'price': float(futures_ob['asks'][0][0]),
                'size': float(futures_ob['asks'][0][1])
            }
           
            return {
                'spot': {'bid': spot_best_bid, 'ask': spot_best_ask},
                'futures': {'bid': futures_best_bid, 'ask': futures_best_ask},
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting orderbook data: {e}")
            return None
   
    def calculate_basis(self, orderbook_data: Dict) -> Tuple[float, float]:
        """
        Calculate basis for both directions
        Long basis: (Futures bid - Spot ask) / Spot ask
        Short basis: (Futures ask - Spot bid) / Spot bid
        """
        # For going long basis (short futures, long spot)
        long_basis = (orderbook_data['futures']['bid']['price'] -
                     orderbook_data['spot']['ask']['price']) / orderbook_data['spot']['ask']['price']
       
        # For going short basis (long futures, short spot)
        short_basis = (orderbook_data['futures']['ask']['price'] -
                      orderbook_data['spot']['bid']['price']) / orderbook_data['spot']['bid']['price']
       
        # Store in history
        self.basis_history.append({
            'timestamp': orderbook_data['timestamp'],
            'long_basis': long_basis,
            'short_basis': short_basis
        })
       
        # Keep only last 1000 entries
        if len(self.basis_history) > 1000:
            self.basis_history = self.basis_history[-1000:]
       
        return long_basis, short_basis
   
    def get_basis_statistics(self) -> Dict:
        """Get current basis statistics"""
        if not self.basis_history:
            return {'current_long_basis': 0, 'current_short_basis': 0}
       
        latest = self.basis_history[-1]
        return {
            'current_long_basis': latest['long_basis'],
            'current_short_basis': latest['short_basis']
        }

class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
   
    def __init__(self, direction: str, basis: float, orderbook_data: Dict):
        self.direction = direction  # 'LONG_BASIS' or 'SHORT_BASIS'
        self.basis = basis
        self.orderbook_data = orderbook_data
        self.timestamp = time.time()
       
        if direction == 'LONG_BASIS':
            # Short futures (market), Long spot (limit)
            self.futures_side = 'SELL'
            self.futures_price = orderbook_data['futures']['bid']['price']
            self.spot_side = 'BUY'
            self.spot_price = orderbook_data['spot']['ask']['price']
            # For limit orders, we want to join the best bid
            self.spot_limit_price = orderbook_data['spot']['bid']['price']
            self.available_liquidity = min(
                orderbook_data['futures']['bid']['size'],
                orderbook_data['spot']['ask']['size']
            )
        else:  # SHORT_BASIS
            # Long futures (market), Short spot (limit)
            self.futures_side = 'BUY'
            self.futures_price = orderbook_data['futures']['ask']['price']
            self.spot_side = 'SELL'
            self.spot_price = orderbook_data['spot']['bid']['price']
            # For limit orders, we want to join the best ask
            self.spot_limit_price = orderbook_data['spot']['ask']['price']
            self.available_liquidity = min(
                orderbook_data['futures']['ask']['size'],
                orderbook_data['spot']['bid']['size']
            )

class ActiveLimitOrder:
    """Track active limit orders and their fills"""
   
    def __init__(self, order_id: str, opportunity: ArbitrageOpportunity,
                 size: float, price: float):
        self.order_id = order_id
        self.opportunity = opportunity
        self.original_size = size
        self.remaining_size = size
        self.filled_size = 0
        self.hedged_size = 0  # How much has been hedged on futures
        self.price = price
        self.last_price_update = time.time()
        self.created_at = time.time()
       
    def update_fill(self, filled_size: float):
        """Update fill information"""
        self.filled_size = filled_size
        self.remaining_size = self.original_size - filled_size
       
    def get_unhedged_size(self) -> float:
        """Get the size that hasn't been hedged yet"""
        return self.filled_size - self.hedged_size
   
    def mark_hedged(self, size: float):
        """Mark size as hedged"""
        self.hedged_size += size

class OrderExecutor:
    """Handle order execution with spot limit, futures market"""
   
    def __init__(self, config: BackpackArbitrageConfig):
        self.config = config
        self.active_limit_orders = {}  # order_id -> ActiveLimitOrder
       
    async def execute_arbitrage(self, client, opportunity: ArbitrageOpportunity,
                               position_manager) -> Optional[Dict]:
        """Execute arbitrage opportunity with spot limit order"""
        try:
            # Calculate order size
            order_size = self._calculate_order_size(opportunity, position_manager)
            if order_size <= 0:
                logger.info("Order size too small or position limit reached")
                return None
           
            # Place limit order on spot
            spot_order = await self._place_spot_limit_order(
                client, opportunity, order_size
            )
           
            if not spot_order:
                return None
           
            # Create active order tracking
            active_order = ActiveLimitOrder(
                order_id=spot_order['id'],
                opportunity=opportunity,
                size=order_size,
                price=opportunity.spot_limit_price
            )
           
            self.active_limit_orders[spot_order['id']] = active_order
           
            return {
                'spot_order': spot_order,
                'opportunity': opportunity,
                'status': 'limit_placed'
            }
           
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return None
   
    async def monitor_and_hedge_fills(self, client, position_manager):
        """Monitor all active limit orders and hedge any new fills"""
        print("Monitoring active limit orders for fills and hedging")
        orders_to_remove = []
       
        for order_id, active_order in self.active_limit_orders.items():
            try:
                # Get current order status
                print("get order")
                order_status = await client.get_order(order_id)
                print("got order")
               
                # Update fill information
                filled_quantity = float(order_status.get('executed_quantity', 0))
                active_order.update_fill(filled_quantity)
               
                # Check if there's unhedged fill
                unhedged_size = active_order.get_unhedged_size()
               
                if unhedged_size >= self.config.min_fill_size:
                    # Execute futures market order to hedge
                    logger.info(f"Hedging {unhedged_size:.3f} SOL from order {order_id}")
                   
                    futures_order = await client.create_market_order(
                        symbol=self.config.futures_symbol,
                        side=active_order.opportunity.futures_side,
                        quantity=unhedged_size
                    )
                   
                    # Mark as hedged
                    active_order.mark_hedged(unhedged_size)
                   
                    # Update position
                    position_manager.update_position({
                        'spot_fill': unhedged_size,
                        'futures_fill': unhedged_size,
                        'opportunity': active_order.opportunity,
                        'spot_price': active_order.price,
                        'futures_price': futures_order.get('price', active_order.opportunity.futures_price)
                    })
               
                # Check if order is completed or cancelled
                if order_status['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    orders_to_remove.append(order_id)
                    logger.info(f"Order {order_id} completed with status {order_status['status']}")
               
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
       
        # Remove completed orders
        for order_id in orders_to_remove:
            del self.active_limit_orders[order_id]
   
    async def update_limit_order_prices(self,public_client, client, market_analyzer: MarketDataAnalyzer):
        """Update prices of active limit orders based on current market"""
        current_time = time.time()
       
        for order_id, active_order in list(self.active_limit_orders.items()):
            try:
                # Check if it's time to update price
                if current_time - active_order.last_price_update < self.config.price_update_interval:
                    continue
               
                # Skip if order has partial fills (to avoid complications)
                if active_order.filled_size > 0:
                    continue
               
                # Get current orderbook
                orderbook_data = await market_analyzer.get_orderbook_data(public_client,client)
                if not orderbook_data:
                    continue
               
                # Calculate new price based on current orderbook
                if active_order.opportunity.spot_side == 'BUY':
                    new_price = orderbook_data['spot']['bid']['price']
                else:
                    new_price = orderbook_data['spot']['ask']['price']
               
                # Check if price has changed significantly (more than 0.01%)
                price_change = abs(new_price - active_order.price) / active_order.price
               
                if price_change > 0.0001:
                    # Cancel old order
                    await client.cancel_order(order_id)
                   
                    # Place new order with updated price
                    new_opportunity = ArbitrageOpportunity(
                        direction=active_order.opportunity.direction,
                        basis=active_order.opportunity.basis,
                        orderbook_data=orderbook_data
                    )
                   
                    new_order = await self._place_spot_limit_order(
                        client, new_opportunity, active_order.remaining_size
                    )
                   
                    if new_order:
                        # Remove old tracking and add new
                        del self.active_limit_orders[order_id]
                       
                        new_active_order = ActiveLimitOrder(
                            order_id=new_order['id'],
                            opportunity=new_opportunity,
                            size=active_order.remaining_size,
                            price=new_price
                        )
                       
                        self.active_limit_orders[new_order['id']] = new_active_order
                        logger.info(f"Updated limit order price from ${active_order.price:.2f} to ${new_price:.2f}")
               
            except Exception as e:
                logger.error(f"Error updating order {order_id}: {e}")
   
    def _calculate_order_size(self, opportunity: ArbitrageOpportunity,
                             position_manager) -> float:
        """Calculate appropriate order size in SOL"""
        # Get remaining position capacity
        remaining_capacity = position_manager.get_remaining_capacity()
       
        # Convert to SOL
        sol_price = opportunity.spot_price
        max_sol_capacity = remaining_capacity / sol_price
       
        # Consider available liquidity
        max_sol_liquidity = opportunity.available_liquidity
       
        # Consider max order size
        max_sol_order = self.config.max_order_size / sol_price
       
        # Take minimum of all constraints
        order_size = min(max_sol_capacity, max_sol_liquidity, max_sol_order)
       
        # Round to reasonable precision
        return round(order_size, 3)
   
    async def _place_spot_limit_order(self, client, opportunity: ArbitrageOpportunity,
                                     order_size: float) -> Optional[Dict]:
        """Place limit order on spot"""
        try:
            order = await client.create_limit_order(
                symbol=self.config.spot_symbol,
                side=opportunity.spot_side,
                price=opportunity.spot_limit_price,
                quantity=order_size,
                post_only=True  # Ensure maker fee (0%)
            )
           
            logger.info(f"Placed spot {opportunity.spot_side} limit order: "
                       f"{order_size} SOL @ ${opportunity.spot_limit_price}")
           
            return order
           
        except Exception as e:
            logger.error(f"Error placing spot order: {e}")
            return None
   
    async def cancel_all_orders(self, client):
        """Cancel all active limit orders"""
        for order_id in list(self.active_limit_orders.keys()):
            try:
                await client.cancel_order(order_id)
                logger.info(f"Cancelled order {order_id}")
            except:
                pass
        self.active_limit_orders.clear()

class PositionManager:
    """Manage positions and risk"""
   
    def __init__(self, config: BackpackArbitrageConfig):
        self.config = config
        self.positions = {
            'spot': 0,
            'futures': 0
        }
        self.trades = []
        self.funding_count = 0
        self.position_start_time = None
       
    def update_position(self, trade_info: Dict):
        """Update positions after trade execution"""
        quantity = trade_info['spot_fill']
       
        if trade_info['opportunity'].direction == 'LONG_BASIS':
            # Short futures, Long spot
            self.positions['futures'] -= quantity
            self.positions['spot'] += quantity
        else:  # SHORT_BASIS
            # Long futures, Short spot
            self.positions['futures'] += quantity
            self.positions['spot'] -= quantity
       
        # Record trade
        self.trades.append({
            'timestamp': time.time(),
            'direction': trade_info['opportunity'].direction,
            'quantity': quantity,
            'basis': trade_info['opportunity'].basis,
            'spot_price': trade_info['spot_price'],
            'futures_price': trade_info['futures_price']
        })
       
        # Set position start time if first trade
        if self.position_start_time is None:
            self.position_start_time = time.time()
       
        logger.info(f"Position updated - Spot: {self.positions['spot']:.3f}, "
                   f"Futures: {self.positions['futures']:.3f}")
   
    def get_net_position(self) -> float:
        """Get net position (should be close to 0 for hedged positions)"""
        return self.positions['spot'] + self.positions['futures']
   
    def get_basis_position(self) -> float:
        """Get basis position size (absolute value of spot position)"""
        return abs(self.positions['spot'])
   
    def get_remaining_capacity(self) -> float:
        """Get remaining position capacity in USD"""
        current_position_value = self.get_basis_position() * self._get_current_sol_price()
        return self.config.max_position_limit - current_position_value
   
    def should_close_positions(self, current_basis_stats: Dict) -> Tuple[bool, str]:
        """Determine if positions should be closed"""
        if self.get_basis_position() < 0.001:  # No significant position
            return False, ""
       
        # Check funding periods (8 hours each)
        if self.position_start_time:
            hours_elapsed = (time.time() - self.position_start_time) / 3600
            funding_periods = int(hours_elapsed / 8)
           
            if funding_periods >= self.config.max_funding_periods:
                return True, "max_funding_periods"
       
        # Check if basis has reversed significantly
        current_long_basis = current_basis_stats.get('current_long_basis', 0)
        current_short_basis = current_basis_stats.get('current_short_basis', 0)
       
        # If we're long basis and it's now negative
        if self.positions['spot'] > 0 and current_long_basis < 0:
            return True, "basis_reversed"
       
        # If we're short basis and it's now positive
        if self.positions['spot'] < 0 and current_short_basis > 0:
            return True, "basis_reversed"
       
        # Check for extreme basis (risk management)
        if abs(current_long_basis) > self.config.extreme_basis_close:
            return True, "extreme_basis"
       
        return False, ""
   
    def _get_current_sol_price(self) -> float:
        """Get current SOL price (simplified - should use real market data)"""
        return 140.0  # Placeholder
   
    def calculate_pnl(self) -> Dict:
        """Calculate current PnL"""
        total_volume = sum(t['quantity'] * self._get_current_sol_price() for t in self.trades)
       
        # Calculate captured basis value
        basis_pnl = sum(t['quantity'] * t['basis'] * self._get_current_sol_price()
                       for t in self.trades)
       
        # Estimate fees (0.05% on futures market orders)
        total_fees = total_volume * 0.0005
       
        net_pnl = basis_pnl - total_fees
       
        return {
            'total_volume': total_volume,
            'basis_pnl': basis_pnl,
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'trade_count': len(self.trades)
        }

class BackpackArbitrageBot:
    """Main arbitrage bot"""
   
    def __init__(self, public_client, client):
        self.public_client = public_client;
        self.client = client
        self.config = BackpackArbitrageConfig()
        self.market_analyzer = MarketDataAnalyzer(self.config)
        self.order_executor = OrderExecutor(self.config)
        self.position_manager = PositionManager(self.config)
        self.running = False
       
    async def start(self):
        """Start the arbitrage bot"""
        self.running = True
        logger.info("Starting Backpack SOL Spot-Futures Arbitrage Bot (Spot Limit/Futures Market)")
       
        last_opportunity_check = 0
       
        while self.running:
            try:
                current_time = time.time()
               
                # Always monitor and hedge fills (high frequency)
                await self.order_executor.monitor_and_hedge_fills(
                    self.client, self.position_manager
                )
               
                # Update limit order prices periodically
                await self.order_executor.update_limit_order_prices(
                    self.public_client,self.client, self.market_analyzer
                )
               
                # Check for new opportunities every 1 second
                if current_time - last_opportunity_check >= 1:
                    last_opportunity_check = current_time
                   
                    # Check if we should close positions
                    basis_stats = self.market_analyzer.get_basis_statistics()
                    should_close, reason = self.position_manager.should_close_positions(basis_stats)
                   
                    if should_close:
                        logger.info(f"Closing positions due to: {reason}")
                        await self._close_all_positions()
                        continue
                   
                    # Get market data
                    orderbook_data = await self.market_analyzer.get_orderbook_data(self.public_client,self.client)
                    if not orderbook_data:
                        continue
                   
                    # Calculate basis
                    long_basis, short_basis = self.market_analyzer.calculate_basis(orderbook_data)
                   
                    # Check for arbitrage opportunities
                    opportunities = []
                   
                    # Long basis opportunity (short futures, long spot)
                    if long_basis > self.config.target_basis_long + self.config.min_basis_threshold:
                        opportunities.append(ArbitrageOpportunity(
                            direction='LONG_BASIS',
                            basis=long_basis,
                            orderbook_data=orderbook_data
                        ))
                   
                    # Short basis opportunity (long futures, short spot)
                    elif short_basis < self.config.target_basis_short - self.config.min_basis_threshold:
                        opportunities.append(ArbitrageOpportunity(
                            direction='SHORT_BASIS',
                            basis=short_basis,
                            orderbook_data=orderbook_data
                        ))
                   
                    # Execute opportunities
                    for opportunity in opportunities:
                        if self.position_manager.get_remaining_capacity() <= 0:
                            logger.info("Position limit reached")
                            break
                       
                        # Check if we already have too many active orders
                        if len(self.order_executor.active_limit_orders) >= 5:
                            logger.info("Too many active orders")
                            break
                       
                        logger.info(f"Placing {opportunity.direction} limit order, basis: {opportunity.basis:.4%}")
                       
                        result = await self.order_executor.execute_arbitrage(
                            self.client, opportunity, self.position_manager
                        )
                       
                        if result:
                            self._display_dashboard()
               
                await asyncio.sleep(0.05)  # 50ms loop for responsive fill monitoring
               
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
   
    async def _close_all_positions(self):
        """Close all positions"""
        try:
            # First cancel all active orders
            await self.order_executor.cancel_all_orders(self.client)
           
            # Then close any remaining positions
            net_position = self.position_manager.get_net_position()
           
            if abs(net_position) > 0.001:
                # We have unhedged position, close it
                if net_position > 0:
                    # Net long, need to sell
                    await self.client.create_market_order(
                        symbol=self.config.spot_symbol,
                        side='SELL',
                        quantity=abs(net_position)
                    )
                else:
                    # Net short, need to buy
                    await self.client.create_market_order(
                        symbol=self.config.spot_symbol,
                        side='BUY',
                        quantity=abs(net_position)
                    )
               
                logger.info(f"Closed net position of {net_position:.3f} SOL")
           
            # Reset positions
            self.position_manager.positions = {'spot': 0, 'futures': 0}
            self.position_manager.position_start_time = None
           
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
   
    def _display_dashboard(self):
        """Display current status"""
        basis_stats = self.market_analyzer.get_basis_statistics()
        pnl_stats = self.position_manager.calculate_pnl()
       
        active_orders_info = []
        for order_id, active_order in self.order_executor.active_limit_orders.items():
            active_orders_info.append(
                f"  {active_order.opportunity.spot_side} {active_order.original_size:.3f} SOL "
                f"@ ${active_order.price:.2f} (filled: {active_order.filled_size:.3f}, "
                f"hedged: {active_order.hedged_size:.3f})"
            )
       
        logger.info(f"""
        ===== Backpack SOL Arbitrage Dashboard =====
        Current Basis - Long: {basis_stats.get('current_long_basis', 0):.4%} | Short: {basis_stats.get('current_short_basis', 0):.4%}
        Position - Spot: {self.position_manager.positions['spot']:.3f} | Futures: {self.position_manager.positions['futures']:.3f}
       
        Active Limit Orders ({len(self.order_executor.active_limit_orders)}):
{chr(10).join(active_orders_info) if active_orders_info else "  None"}
       
        Volume Generated: ${pnl_stats['total_volume']:,.2f}
        Net PnL: ${pnl_stats['net_pnl']:,.2f}
        Trades: {pnl_stats['trade_count']}
        Capacity Used: {(self.config.max_position_limit - self.position_manager.get_remaining_capacity()) / self.config.max_position_limit:.1%}
        ==========================================
        """)
   
    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Stopping arbitrage bot")



async def main():

    from backpack_exchange_sdk.public import PublicClient

    public_client = PublicClient()

    from backpack_exchange_sdk.authenticated import AuthenticationClient
    client = AuthenticationClient('', '')
   
    bot = BackpackArbitrageBot(public_client,client)
   
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Bot stopped by user")

if __name__ == "__main__":
    print("Backpack SOL Arbitrage Bot")
    print("==========================")
    print("Make sure to set your API credentials in the main() function.")
    print("Press Ctrl+C to stop the bot.\n")
   
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")