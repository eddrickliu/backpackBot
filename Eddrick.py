#!/usr/bin/env python3
"""
Backpack SOL Spot-Futures Arbitrage System (Spot Limit, Futures Market)
FINAL FIXED VERSION - Correct orderbook parsing
"""

import time
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from enums import RequestEnums
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackpackArbitrageConfig:
    """Configuration for the arbitrage system"""
    def __init__(self):
        # Trading pairs - ONLY TWO NEEDED!
        self.spot_symbol = "SOL_USDC"      # For spot market
        self.futures_symbol = "SOL_USDC_PERP"   # For futures/perp market
        
        # Risk parameters
        self.max_position_limit = 50      # USD value
        self.max_order_size = 1         # USD per order
        
        # Adjusted thresholds for testing
        self.min_basis_threshold = 0.0001    # 0.01% (lowered for testing)
        
        # Based on historical data analysis
        self.target_basis_long = 0.0001      # 0.01% - lowered for testing
        self.target_basis_short = -0.0001    # -0.01% - lowered for testing
        self.extreme_basis_close = 0.01      # 1% - close positions when basis is extreme
        
        # Position management
        self.max_funding_periods = 3         # Close after 3 funding periods (24 hours)
        self.emergency_close_loss = 100      # USD loss trigger
        
        # Order execution
        self.limit_order_timeout = 30        # seconds
        self.order_check_interval = 0.1      # seconds
        self.price_update_interval = 2       # seconds - how often to update limit order prices
        self.min_fill_size = 0.001          # Minimum fill size to trigger hedge

async def debug_orderbook_format(public_client):
    """Debug function to understand Backpack's orderbook format"""
    print("\n=== Debugging Orderbook Format ===")
    
    try:
        # Get spot orderbook
        spot_ob = public_client.get_depth("SOL_USDC")
        
        print("\nRaw Spot Orderbook Data:")
        print(f"Number of bids: {len(spot_ob['bids'])}")
        print(f"Number of asks: {len(spot_ob['asks'])}")
        
        # Show first 5 bids
        print("\nFirst 5 Bids:")
        for i, bid in enumerate(spot_ob['bids'][:5]):
            print(f"  Bid {i}: {bid}")
            
        # Show last 5 bids (in case they're reversed)
        if len(spot_ob['bids']) > 5:
            print("\nLast 5 Bids:")
            for i, bid in enumerate(spot_ob['bids'][-5:]):
                print(f"  Bid {len(spot_ob['bids'])-5+i}: {bid}")
        
        # Analyze which field is likely price
        print("\n=== Analyzing Format ===")
        # Check if values in position 0 or 1 are more likely to be prices
        position_0_values = [float(bid[0]) for bid in spot_ob['bids'][:5]]
        position_1_values = [float(bid[1]) for bid in spot_ob['bids'][:5]]
        
        print(f"Position [0] values: {position_0_values}")
        print(f"Position [1] values: {position_1_values}")
        
        # SOL price should be between $20 and $500
        pos0_likely_price = all(20 < v < 500 for v in position_0_values)
        pos1_likely_price = all(20 < v < 500 for v in position_1_values)
        
        if pos0_likely_price and not pos1_likely_price:
            print("✓ Format appears to be [price, size]")
            format_type = "price_size"
        elif pos1_likely_price and not pos0_likely_price:
            print("✓ Format appears to be [size, price]")
            format_type = "size_price"
        else:
            print("⚠️  Cannot determine format automatically")
            format_type = "unknown"
            
        # Check sort order
        if format_type == "price_size":
            first_price = float(spot_ob['bids'][0][0])
            last_price = float(spot_ob['bids'][-1][0])
        elif format_type == "size_price":
            first_price = float(spot_ob['bids'][0][1])
            last_price = float(spot_ob['bids'][-1][1])
        else:
            first_price = float(spot_ob['bids'][0][0])
            last_price = float(spot_ob['bids'][-1][0])
            
        if first_price > last_price:
            print("✓ Bids are sorted descending (highest first) - STANDARD")
        else:
            print("⚠️  Bids are sorted ascending (lowest first) - UNUSUAL")
            
        return format_type, first_price > last_price
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return "unknown", True

async def discover_backpack_symbols(public_client):
    """Discover available symbols on Backpack exchange"""
    print("\n=== Discovering Backpack Symbols ===")
    
    try:
        # Common symbol formats to try for SOL perpetuals
        potential_futures_symbols = [
            # "SOL_PERP",
            # "SOL-PERP", 
            "SOL_USDC_PERP",
            # "SOL-USDC-PERP",
            # "SOLUSDT",
            # "SOL-USDT-PERP",
            # "SOL_USDT_PERP",
            # "SOL/USD",
            # "SOL_USD"
        ]
        
        print("\nTesting potential futures symbols:")
        working_futures_symbol = None
        
        for symbol in potential_futures_symbols:
            try:
                result = public_client.get_depth(symbol)
                if result and 'bids' in result and len(result['bids']) > 0:
                    # Quick check if price looks reasonable
                    val0 = float(result['bids'][0][0])
                    val1 = float(result['bids'][0][1])
                    price = val0 if 20 < val0 < 500 else val1
                    print(f"✓ {symbol} - WORKS! Estimated price: ${price:.2f}")
                    working_futures_symbol = symbol
                    break
            except Exception as e:
                print(f"✗ {symbol} - Failed")
                print(f"✗ Error: {e}")
        
        # Test spot symbol
        print("\nTesting spot symbol:")
        try:
            spot_result = public_client.get_depth("SOL_USDC")
            if spot_result and 'bids' in spot_result:
                val0 = float(spot_result['bids'][0][0])
                val1 = float(spot_result['bids'][0][1])
                price = val0 if 20 < val0 < 500 else val1
                print(f"✓ SOL_USDC - WORKS! Estimated price: ${price:.2f}")
            else:
                print("✗ SOL_USDC - Failed")
        except:
            print("✗ SOL_USDC - Failed")
            
        if working_futures_symbol:
            print(f"\n✓ Found working futures symbol: {working_futures_symbol}")
            return "SOL_USDC", working_futures_symbol
        else:
            print("\n✗ Could not find working futures symbol")
            return "SOL_USDC", None
            
    except Exception as e:
        print(f"Error discovering symbols: {e}")
        return "SOL_USDC", None

class MarketDataAnalyzer:
    """Analyze market data and calculate basis"""
    
    def __init__(self, config: BackpackArbitrageConfig):
        self.config = config
        self.basis_history = []
        self.last_orderbook_data = None
        self.orderbook_format = None  # Will be determined on first fetch
        self.bids_descending = True   # Will be determined on first fetch
        
    def _parse_orderbook_entry(self, entry, format_type="auto"):
        """Parse orderbook entry based on detected format"""
        val0 = float(entry[0])
        val1 = float(entry[1])
        
        if format_type == "auto":
            # Auto-detect based on reasonable price range for SOL
            if 20 < val0 < 500:  # val0 looks like price
                return {'price': val0, 'size': val1}
            elif 20 < val1 < 500:  # val1 looks like price
                return {'price': val1, 'size': val0}
            else:
                # Default assumption
                return {'price': val0, 'size': val1}
        elif format_type == "price_size":
            return {'price': val0, 'size': val1}
        elif format_type == "size_price":
            return {'price': val1, 'size': val0}
        else:
            return {'price': val0, 'size': val1}
        
    async def get_orderbook_data(self, public_client, client) -> Dict:
        """Get orderbook data for both spot and futures - FIXED VERSION"""
        try:
            # Get spot orderbook
            logger.debug(f"Fetching spot orderbook for {self.config.spot_symbol}")
            spot_ob = public_client.get_depth(self.config.spot_symbol)
            
            # Get futures orderbook
            logger.debug(f"Fetching futures orderbook for {self.config.futures_symbol}")
            futures_ob = public_client.get_depth(self.config.futures_symbol)
            
            # Validate orderbook data
            if not spot_ob or 'bids' not in spot_ob or 'asks' not in spot_ob:
                logger.error("Invalid spot orderbook data")
                return None
                
            if not futures_ob or 'bids' not in futures_ob or 'asks' not in futures_ob:
                logger.error("Invalid futures orderbook data")
                return None
                
            # Check if we have any bids/asks
            if len(spot_ob['bids']) == 0 or len(spot_ob['asks']) == 0:
                logger.error("Empty spot orderbook")
                return None
                
            if len(futures_ob['bids']) == 0 or len(futures_ob['asks']) == 0:
                logger.error("Empty futures orderbook")
                return None
            
            # Detect format on first run
            if self.orderbook_format is None:
                # Check multiple entries to determine format
                spot_prices_pos0 = [float(bid[0]) for bid in spot_ob['bids'][:min(5, len(spot_ob['bids']))]]
                spot_prices_pos1 = [float(bid[1]) for bid in spot_ob['bids'][:min(5, len(spot_ob['bids']))]]
                
                # Check which position has price-like values
                pos0_likely = sum(1 for p in spot_prices_pos0 if 20 < p < 500) >= 3
                pos1_likely = sum(1 for p in spot_prices_pos1 if 20 < p < 500) >= 3
                
                if pos0_likely and not pos1_likely:
                    self.orderbook_format = "price_size"
                    logger.info("Detected orderbook format: [price, size]")
                elif pos1_likely and not pos0_likely:
                    self.orderbook_format = "size_price"
                    logger.info("Detected orderbook format: [size, price]")
                else:
                    self.orderbook_format = "price_size"  # Default
                    logger.warning("Could not detect orderbook format, using default [price, size]")
                
                # Check sort order
                if self.orderbook_format == "price_size":
                    first_price = float(spot_ob['bids'][0][0])
                    last_price = float(spot_ob['bids'][-1][0])
                else:
                    first_price = float(spot_ob['bids'][0][1])
                    last_price = float(spot_ob['bids'][-1][1])
                
                self.bids_descending = first_price > last_price
                logger.info(f"Bids sorted: {'descending' if self.bids_descending else 'ascending'}")
            
            # Parse best bid/ask based on detected format and sort order
            if self.bids_descending:
                # Standard: best bid is first
                spot_best_bid = self._parse_orderbook_entry(spot_ob['bids'][0], self.orderbook_format)
                futures_best_bid = self._parse_orderbook_entry(futures_ob['bids'][0], self.orderbook_format)
            else:
                # Unusual: best bid is last
                spot_best_bid = self._parse_orderbook_entry(spot_ob['bids'][-1], self.orderbook_format)
                futures_best_bid = self._parse_orderbook_entry(futures_ob['bids'][-1], self.orderbook_format)
            
            # Best ask is always first (lowest price)
            spot_best_ask = self._parse_orderbook_entry(spot_ob['asks'][0], self.orderbook_format)
            futures_best_ask = self._parse_orderbook_entry(futures_ob['asks'][0], self.orderbook_format)
            
            # Validate prices are reasonable
            if not (20 < spot_best_bid['price'] < 500):
                logger.error(f"Unreasonable spot bid price: ${spot_best_bid['price']}")
                return None
            if not (20 < spot_best_ask['price'] < 500):
                logger.error(f"Unreasonable spot ask price: ${spot_best_ask['price']}")
                return None
            
            # Log the current prices
            logger.debug(f"Spot: Bid=${spot_best_bid['price']:.2f}, Ask=${spot_best_ask['price']:.2f}")
            logger.debug(f"Futures: Bid=${futures_best_bid['price']:.2f}, Ask=${futures_best_ask['price']:.2f}")
            
            result = {
                'spot': {'bid': spot_best_bid, 'ask': spot_best_ask},
                'futures': {'bid': futures_best_bid, 'ask': futures_best_ask},
                'timestamp': time.time()
            }
            
            self.last_orderbook_data = result
            return result
            
        except Exception as e:
            logger.error(f"Error getting orderbook data: {e}")
            import traceback
            traceback.print_exc()
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
            'short_basis': short_basis,
            'spot_price': orderbook_data['spot']['ask']['price']  # Store spot price
        })
        
        # Keep only last 1000 entries
        if len(self.basis_history) > 1000:
            self.basis_history = self.basis_history[-1000:]
        
        # Log basis for debugging
        logger.debug(f"Calculated basis - Long: {long_basis:.4%}, Short: {short_basis:.4%}")
        
        return long_basis, short_basis
    
    def get_basis_statistics(self) -> Dict:
        """Get current basis statistics"""
        if not self.basis_history:
            return {'current_long_basis': 0, 'current_short_basis': 0, 'current_spot_price': 100.0}
        
        latest = self.basis_history[-1]
        return {
            'current_long_basis': latest['long_basis'],
            'current_short_basis': latest['short_basis'],
            'current_spot_price': latest.get('spot_price', 100.0)
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
            self.spot_side = RequestEnums.Side.BID.value
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
            self.spot_side = RequestEnums.Side.ASK.value
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
        orders_to_remove = []
        
        # print("self.active_limit_orders.items()",self.active_limit_orders.items());
        # print("self.active_limit_orders",self.active_limit_orders);
        for order_id, active_order in self.active_limit_orders.items():
            try:
                # Get current order status
                # order_status = await client.get_users_open_orders("SOL_USDC",int(order_id))
                order_status_s = client.get_open_orders("SOL_USDC")
                # order_status = client.get_users_open_orders("SOL_USDC",orderId=order_id);
                # print("order_statuss",order_status_s);
                matches = [item for item in order_status_s if item["id"] == order_id];
                # print("matches",matches);
                if(matches.__len__() != 0):
                                    
                    # Update fill information
                    filled_quantity = float(matches[0].get("executedQuantity")) 
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
                    if matches[0].get('status') in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        orders_to_remove.append(order_id)
                        logger.info(f"Order {order_id} completed with status { matches[0].get('status')}")

                
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
        
        # Remove completed orders
        for order_id in orders_to_remove:
            del self.active_limit_orders[order_id]
    
    async def update_limit_order_prices(self, public_client, client, market_analyzer: MarketDataAnalyzer):
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
                orderbook_data = await market_analyzer.get_orderbook_data(public_client, client)
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
                    print( "cancelorder_id",order_id);
                    await client.cancel_open_order("SOL_USDC",int(order_id))
                    # del self.active_limit_orders[order_id]

                    print( "cancelorder_id",order_id);
                    
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
            # order = await client.create_limit_order(
            #     symbol=self.config.spot_symbol,
            #     side=opportunity.spot_side, 
            #     price=opportunity.spot_limit_price,
            #     quantity=order_size,
            #     post_only=True  # Ensure maker fee (0%)
            # )
            print(f"Placing spot limit order: {opportunity.spot_side} {order_size} SOL @ ${opportunity.spot_limit_price}")
            rounded_quantity = round(order_size, 2)
            order = client.execute_order(
                RequestEnums.OrderType.LIMIT.value,
                opportunity.spot_side,
                self.config.spot_symbol,
                postOnly=True,  # Ensure maker fee (0%)
                price=opportunity.spot_limit_price,
                quantity=rounded_quantity,
                autoBorrow=True,
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
                await client.cancel_open_order("SOL_USDC",order_id)
                logger.info(f"Cancelled order {order_id}")
            except:
                pass
        self.active_limit_orders.clear()

class PositionManager:
    """Manage positions and risk"""
    
    def __init__(self, config: BackpackArbitrageConfig, market_analyzer: MarketDataAnalyzer):
        self.config = config
        self.market_analyzer = market_analyzer
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
        """Get current SOL price from market data"""
        basis_stats = self.market_analyzer.get_basis_statistics()
        return basis_stats.get('current_spot_price', 100.0)
    
    def calculate_pnl(self) -> Dict:
        """Calculate current PnL"""
        sol_price = self._get_current_sol_price()
        total_volume = sum(t['quantity'] * sol_price for t in self.trades)
        
        # Calculate captured basis value
        basis_pnl = sum(t['quantity'] * t['basis'] * sol_price
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
        self.public_client = public_client
        self.client = client
        self.config = BackpackArbitrageConfig()
        self.market_analyzer = MarketDataAnalyzer(self.config)
        self.order_executor = OrderExecutor(self.config)
        self.position_manager = PositionManager(self.config, self.market_analyzer)
        self.running = False
        
    async def start(self):
        """Start the arbitrage bot"""
        self.running = True
        logger.info("Starting Backpack SOL Spot-Futures Arbitrage Bot (Spot Limit/Futures Market)")
        logger.info(f"Configuration:")
        logger.info(f"  Spot Symbol: {self.config.spot_symbol}")
        logger.info(f"  Futures Symbol: {self.config.futures_symbol}")
        logger.info(f"  Min Basis Threshold: {self.config.min_basis_threshold:.4%}")
        logger.info(f"  Target Long Basis: {self.config.target_basis_long:.4%}")
        logger.info(f"  Target Short Basis: {self.config.target_basis_short:.4%}")
        
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
                    self.public_client, self.client, self.market_analyzer
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
                    orderbook_data = await self.market_analyzer.get_orderbook_data(self.public_client, self.client)
                    if not orderbook_data:
                        logger.debug("No orderbook data available")
                        continue
                    
                    # Calculate basis
                    long_basis, short_basis = self.market_analyzer.calculate_basis(orderbook_data)
                    
                    # Log current market status
                    logger.info(f"Market Status - Spot: Bid=${orderbook_data['spot']['bid']['price']:.2f}, "
                               f"Ask=${orderbook_data['spot']['ask']['price']:.2f} | "
                               f"Futures: Bid=${orderbook_data['futures']['bid']['price']:.2f}, "
                               f"Ask=${orderbook_data['futures']['ask']['price']:.2f}")
                    logger.info(f"Current Basis - Long: {long_basis:.4%}, Short: {short_basis:.4%}")
                    
                    # Calculate spread between spot and futures for additional info
                    spot_mid = (orderbook_data['spot']['bid']['price'] + orderbook_data['spot']['ask']['price']) / 2
                    futures_mid = (orderbook_data['futures']['bid']['price'] + orderbook_data['futures']['ask']['price']) / 2
                    raw_spread = futures_mid - spot_mid
                    raw_spread_pct = (raw_spread / spot_mid) * 100
                    logger.info(f"Spot-Futures Spread: ${raw_spread:.2f} ({raw_spread_pct:.2f}%)")
                    
                    # Check for arbitrage opportunities
                    opportunities = []
                    
                    # Long basis opportunity (short futures, long spot)
                    long_threshold = self.config.target_basis_long + self.config.min_basis_threshold
                    if long_basis > long_threshold:
                        logger.info(f"LONG BASIS OPPORTUNITY DETECTED: {long_basis:.4%} > {long_threshold:.4%}")
                        opportunities.append(ArbitrageOpportunity(
                            direction='LONG_BASIS',
                            basis=long_basis,
                            orderbook_data=orderbook_data
                        ))
                    
                    # Short basis opportunity (long futures, short spot)
                    short_threshold = self.config.target_basis_short - self.config.min_basis_threshold
                    if short_basis < short_threshold:
                        logger.info(f"SHORT BASIS OPPORTUNITY DETECTED: {short_basis:.4%} < {short_threshold:.4%}")
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
                        
                        logger.info(f"Executing {opportunity.direction} opportunity, basis: {opportunity.basis:.4%}")
                        
                        result = await self.order_executor.execute_arbitrage(
                            self.client, opportunity, self.position_manager
                        )
                        
                        if result:
                            self._display_dashboard()
                
                await asyncio.sleep(0.05)  # 50ms loop for responsive fill monitoring
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
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
        Current SOL Price: ${basis_stats.get('current_spot_price', 0):.2f}
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
    # Initialize clients
    from backpack_exchange_sdk.public import PublicClient
    from backpack_exchange_sdk.authenticated import AuthenticationClient
    
    public_client = PublicClient()
    
    # First debug the orderbook format
    print("\n=== Step 1: Understanding Orderbook Format ===")
    format_type, bids_descending = await debug_orderbook_format(public_client)
    
    # Then discover the correct symbols
    print("\n=== Step 2: Discovering Symbols ===")
    spot_symbol, futures_symbol = await discover_backpack_symbols(public_client)
    
    if not futures_symbol:
        print("\n⚠️  WARNING: Could not find a working futures symbol!")
        print("The bot needs both spot and futures symbols to work.")
        print("Please check Backpack's documentation for the correct futures symbol.")
        return
    
    
    API_KEY = ""
    API_SECRET = ""
    
    client = AuthenticationClient(API_KEY, API_SECRET)
    
    # Create and start the bot
    bot = BackpackArbitrageBot(public_client, client)
    
    # Update the config with discovered symbols
    bot.config.spot_symbol = spot_symbol
    bot.config.futures_symbol = futures_symbol
    
    print(f"\n✓ Bot configured with:")
    print(f"  Spot Symbol: {spot_symbol}")
    print(f"  Futures Symbol: {futures_symbol}")
    print(f"  Orderbook Format: {format_type}")
    print(f"  Bids Sorted: {'Descending' if bids_descending else 'Ascending'}")
    print("\nStarting bot...\n")
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Bot stopped by user")


if __name__ == "__main__":
    print("Backpack SOL Arbitrage Bot - FINAL FIXED VERSION")
    print("================================================")
    print("This version will:")
    print("1. Debug and auto-detect orderbook format")
    print("2. Auto-discover the correct futures symbol")
    print("3. Handle both [price,size] and [size,price] formats")
    print("4. Handle both ascending and descending bid sorting")
    print("\nPress Ctrl+C to stop the bot.\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")