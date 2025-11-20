# Saga Pattern 详解

## 目录
1. [Saga概述](#saga概述)
2. [Saga原理](#saga原理)
3. [两种编排模式](#两种编排模式)
4. [Saga实现策略](#saga实现策略)
5. [Python实现](#python实现)
6. [Go语言实现](#go语言实现)
7. [应用场景分析](#应用场景分析)
8. [最佳实践](#最佳实践)
9. [故障处理](#故障处理)
10. [性能优化](#性能优化)

## Saga概述

### 什么是Saga

Saga模式是一种分布式事务管理策略，通过将分布式事务分解为一系列本地事务，每个事务都有对应的补偿操作来实现最终一致性。与2PC/3PC不同，Saga允许多个参与者并行处理，最终通过补偿操作确保所有参与者达到一致状态。

```python
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import asyncio
import threading
import logging
from abc import ABC, abstractmethod
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

class SagaStepStatus(Enum):
    """Saga步骤状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    TIMEOUT = "timeout"

class SagaStatus(Enum):
    """Saga事务状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    TIMEOUT = "timeout"

class SagaEventType(Enum):
    """Saga事件类型"""
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_COMPENSATED = "step_compensated"
    SAGA_STARTED = "saga_started"
    SAGA_COMPLETED = "saga_completed"
    SAGA_FAILED = "saga_failed"
    SAGA_COMPENSATING = "saga_compensating"
    SAGA_COMPENSATED = "saga_compensated"

@dataclass
class SagaStep:
    """Saga步骤定义"""
    step_id: str
    name: str
    service_name: str
    forward_action: Callable
    compensation_action: Callable
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any = None
    error: str = None
    start_time: float = field(default_factory=time.time)
    end_time: float = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    
    def execute(self) -> bool:
        """执行正向操作"""
        self.status = SagaStepStatus.EXECUTING
        
        try:
            if asyncio.iscoroutinefunction(self.forward_action):
                # 异步函数
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.result = loop.run_until_complete(self.forward_action())
            else:
                # 同步函数
                self.result = self.forward_action()
            
            self.status = SagaStepStatus.COMPLETED
            self.end_time = time.time()
            return True
            
        except Exception as e:
            self.status = SagaStepStatus.FAILED
            self.error = str(e)
            self.end_time = time.time()
            return False
    
    def compensate(self) -> bool:
        """执行补偿操作"""
        self.status = SagaStepStatus.COMPENSATING
        
        try:
            if asyncio.iscoroutinefunction(self.compensation_action):
                # 异步函数
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.compensation_action())
            else:
                # 同步函数
                result = self.compensation_action()
            
            self.status = SagaStepStatus.COMPENSATED
            return True
            
        except Exception as e:
            self.error = f"Compensation failed: {str(e)}"
            return False

@dataclass
class SagaEvent:
    """Saga事件"""
    event_type: SagaEventType
    saga_id: str
    step_id: str = None
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

class SagaEventListener(ABC):
    """Saga事件监听器"""
    
    @abstractmethod
    def on_event(self, event: SagaEvent):
        """处理事件"""
        pass

class SagaOrchestrator:
    """Saga编排器"""
    
    def __init__(self, saga_id: str, event_listener: SagaEventListener = None):
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.status = SagaStatus.RUNNING
        self.current_step_index = 0
        self.event_listener = event_listener
        self.execution_log: List[dict] = []
        self.mutex = threading.Lock()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"SagaOrchestrator-{saga_id}")
    
    def add_step(self, step: SagaStep):
        """添加Saga步骤"""
        with self.mutex:
            self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
    
    def execute(self) -> bool:
        """执行Saga"""
        self.status = SagaStatus.RUNNING
        self._emit_event(SagaEventType.SAGA_STARTED, self.saga_id)
        
        self.logger.info(f"Starting saga execution: {self.saga_id}")
        self.logger.info(f"Total steps: {len(self.steps)}")
        
        try:
            # 正向执行所有步骤
            for step in self.steps:
                self.logger.info(f"Executing step: {step.name}")
                
                success = self._execute_step_with_retry(step)
                if not success:
                    self.logger.error(f"Step {step.name} failed, starting compensation")
                    self._compensate()
                    return False
            
            self.status = SagaStatus.COMPLETED
            self._emit_event(SagaEventType.SAGA_COMPLETED, self.saga_id)
            self.logger.info("Saga completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Saga execution failed: {str(e)}")
            self.status = SagaStatus.FAILED
            self._emit_event(SagaEventType.SAGA_FAILED, self.saga_id, {"error": str(e)})
            return False
    
    def _execute_step_with_retry(self, step: SagaStep) -> bool:
        """执行步骤，包含重试逻辑"""
        max_attempts = step.max_retries + 1
        
        for attempt in range(max_attempts):
            step.retry_count = attempt
            
            if attempt > 0:
                self.logger.info(f"Retrying step {step.name}, attempt {attempt + 1}")
                time.sleep(2 ** attempt)  # 指数退避
            
            self._emit_event(SagaEventType.STEP_STARTED, self.saga_id, step.step_id)
            
            success = self._execute_with_timeout(step)
            
            if success:
                self._emit_event(SagaEventType.STEP_COMPLETED, self.saga_id, step.step_id)
                self.logger.info(f"Step {step.name} completed successfully")
                return True
            else:
                self.logger.warning(f"Step {step.name} failed on attempt {attempt + 1}")
                
                if attempt < max_attempts - 1:
                    # 继续重试
                    continue
                else:
                    # 最后一次尝试失败
                    self._emit_event(SagaEventType.STEP_FAILED, self.saga_id, step.step_id, 
                                   {"error": step.error, "attempts": attempt + 1})
                    return False
        
        return False
    
    def _execute_with_timeout(self, step: SagaStep) -> bool:
        """带超时的执行"""
        result_container = [None]
        exception_container = [None]
        
        def target():
            try:
                result_container[0] = step.execute()
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=step.timeout)
        
        if thread.is_alive():
            # 超时
            step.status = SagaStepStatus.TIMEOUT
            step.error = f"Execution timeout after {step.timeout} seconds"
            return False
        
        if exception_container[0]:
            # 执行异常
            step.error = str(exception_container[0])
            return False
        
        return result_container[0] or False
    
    def _compensate(self):
        """执行补偿操作"""
        self.status = SagaStatus.COMPENSATING
        self._emit_event(SagaEventType.SAGA_COMPENSATING, self.saga_id)
        
        self.logger.info("Starting compensation phase")
        
        # 逆序执行补偿操作
        completed_steps = [step for step in self.steps if step.status == SagaStepStatus.COMPLETED]
        compensation_steps = reversed(completed_steps)
        
        compensation_success = True
        
        for step in compensation_steps:
            self.logger.info(f"Compensating step: {step.name}")
            
            success = step.compensate()
            
            if success:
                self._emit_event(SagaEventType.STEP_COMPENSATED, self.saga_id, step.step_id)
                self.logger.info(f"Step {step.name} compensated successfully")
            else:
                compensation_success = False
                self.logger.error(f"Failed to compensate step {step.name}: {step.error}")
                # 继续尝试补偿其他步骤
        
        if compensation_success:
            self.status = SagaStatus.COMPENSATED
            self._emit_event(SagaEventType.SAGA_COMPENSATED, self.saga_id)
            self.logger.info("Saga compensation completed successfully")
        else:
            self.status = SagaStatus.FAILED
            self._emit_event(SagaEventType.SAGA_FAILED, self.saga_id, 
                           {"error": "Compensation partially failed"})
            self.logger.error("Saga compensation partially failed")
    
    def _emit_event(self, event_type: SagaEventType, saga_id: str, 
                   step_id: str = None, data: dict = None):
        """发射事件"""
        if data is None:
            data = {}
        
        event = SagaEvent(
            event_type=event_type,
            saga_id=saga_id,
            step_id=step_id,
            data=data
        )
        
        # 记录事件
        self.execution_log.append({
            "event_type": event_type.value,
            "saga_id": saga_id,
            "step_id": step_id,
            "timestamp": event.timestamp,
            "data": data
        })
        
        # 通知监听器
        if self.event_listener:
            try:
                self.event_listener.on_event(event)
            except Exception as e:
                self.logger.error(f"Error in event listener: {str(e)}")
        
        self.logger.info(f"Event: {event_type.value} - Saga: {saga_id} - Step: {step_id}")
    
    def get_status(self) -> Dict:
        """获取Saga状态"""
        return {
            "saga_id": self.saga_id,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps if s.status == SagaStepStatus.COMPLETED]),
            "failed_steps": len([s for s in self.steps if s.status == SagaStepStatus.FAILED]),
            "execution_log": self.execution_log
        }

class SagaEventLogger(SagaEventListener):
    """Saga事件日志器"""
    
    def __init__(self):
        self.events = []
        self.logger = logging.getLogger("SagaEventLogger")
    
    def on_event(self, event: SagaEvent):
        """记录事件"""
        self.events.append(event)
        self.logger.info(f"Saga Event: {event.event_type.value} - {event.saga_id}")

class ECommerceSagaOrchestrator(SagaOrchestrator):
    """电商Saga编排器"""
    
    def __init__(self, saga_id: str, order_service, inventory_service, payment_service, shipping_service):
        super().__init__(saga_id)
        self.order_service = order_service
        self.inventory_service = inventory_service
        self.payment_service = payment_service
        self.shipping_service = shipping_service
    
    def create_order_saga(self, order_data: dict):
        """创建订单Saga"""
        
        # 步骤1: 创建订单
        def create_order():
            return self.order_service.create_order(order_data)
        
        def cancel_order():
            if hasattr(self, '_created_order_id'):
                return self.order_service.cancel_order(self._created_order_id)
            return True
        
        step1 = SagaStep(
            step_id="create_order",
            name="Create Order",
            service_name="order_service",
            forward_action=create_order,
            compensation_action=cancel_order
        )
        
        # 步骤2: 预留库存
        def reserve_inventory():
            items = order_data.get("items", [])
            self._inventory_reservations = self.inventory_service.reserve_inventory(items)
            return self._inventory_reservations
        
        def release_inventory():
            if hasattr(self, '_inventory_reservations'):
                return self.inventory_service.release_reservations(self._inventory_reservations)
            return True
        
        step2 = SagaStep(
            step_id="reserve_inventory",
            name="Reserve Inventory",
            service_name="inventory_service",
            forward_action=reserve_inventory,
            compensation_action=release_inventory
        )
        
        # 步骤3: 处理支付
        def process_payment():
            payment_info = order_data.get("payment", {})
            self._payment_result = self.payment_service.process_payment(payment_info)
            return self._payment_result
        
        def refund_payment():
            if hasattr(self, '_payment_result'):
                return self.payment_service.refund_payment(self._payment_result)
            return True
        
        step3 = SagaStep(
            step_id="process_payment",
            name="Process Payment",
            service_name="payment_service",
            forward_action=process_payment,
            compensation_action=refund_payment
        )
        
        # 步骤4: 安排配送
        def schedule_shipping():
            shipping_info = order_data.get("shipping", {})
            self._shipping_order_id = self.shipping_service.schedule_shipping(shipping_info)
            return self._shipping_order_id
        
        def cancel_shipping():
            if hasattr(self, '_shipping_order_id'):
                return self.shipping_service.cancel_shipping(self._shipping_order_id)
            return True
        
        step4 = SagaStep(
            step_id="schedule_shipping",
            name="Schedule Shipping",
            service_name="shipping_service",
            forward_action=schedule_shipping,
            compensation_action=cancel_shipping
        )
        
        # 添加所有步骤
        self.add_step(step1)
        self.add_step(step2)
        self.add_step(step3)
        self.add_step(step4)

class MockOrderService:
    """模拟订单服务"""
    
    def __init__(self):
        self.orders = {}
        self.logger = logging.getLogger("OrderService")
    
    def create_order(self, order_data: dict) -> dict:
        """创建订单"""
        order_id = f"order_{uuid.uuid4().hex[:8]}"
        order = {
            "order_id": order_id,
            "status": "created",
            "items": order_data.get("items", []),
            "total_amount": order_data.get("total_amount", 0),
            "created_at": time.time()
        }
        
        self.orders[order_id] = order
        self.logger.info(f"Created order: {order_id}")
        
        # 模拟服务响应延迟
        time.sleep(0.5)
        
        return {"order_id": order_id, "status": "created"}
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        return False

class MockInventoryService:
    """模拟库存服务"""
    
    def __init__(self):
        self.inventory = {
            "item1": {"stock": 100, "reserved": 0},
            "item2": {"stock": 50, "reserved": 0},
            "item3": {"stock": 25, "reserved": 0}
        }
        self.reservations = {}
        self.logger = logging.getLogger("InventoryService")
    
    def reserve_inventory(self, items: List[dict]) -> dict:
        """预留库存"""
        reservations = {}
        
        for item in items:
            item_id = item.get("item_id")
            quantity = item.get("quantity", 1)
            
            if item_id in self.inventory:
                # 检查库存
                available_stock = self.inventory[item_id]["stock"] - self.inventory[item_id]["reserved"]
                
                if available_stock >= quantity:
                    self.inventory[item_id]["reserved"] += quantity
                    reservations[item_id] = quantity
                    self.logger.info(f"Reserved {quantity} units of {item_id}")
                else:
                    raise Exception(f"Insufficient stock for {item_id}")
            else:
                raise Exception(f"Item {item_id} not found")
        
        time.sleep(0.3)  # 模拟服务延迟
        return reservations
    
    def release_reservations(self, reservations: dict) -> bool:
        """释放预留"""
        for item_id, quantity in reservations.items():
            if item_id in self.inventory:
                self.inventory[item_id]["reserved"] -= quantity
                self.logger.info(f"Released reservation: {quantity} units of {item_id}")
        
        time.sleep(0.2)  # 模拟服务延迟
        return True

class MockPaymentService:
    """模拟支付服务"""
    
    def __init__(self):
        self.transactions = {}
        self.logger = logging.getLogger("PaymentService")
    
    def process_payment(self, payment_info: dict) -> dict:
        """处理支付"""
        payment_id = f"payment_{uuid.uuid4().hex[:8]}"
        amount = payment_info.get("amount", 0)
        card_number = payment_info.get("card_number", "")
        
        # 模拟支付处理
        self.logger.info(f"Processing payment: ${amount}")
        time.sleep(0.8)  # 模拟支付处理时间
        
        # 模拟5%的失败率
        if amount > 1000:  # 大额支付失败模拟
            raise Exception("Payment failed: Insufficient funds")
        
        transaction = {
            "payment_id": payment_id,
            "amount": amount,
            "status": "completed",
            "card_last4": card_number[-4:] if card_number else "****",
            "processed_at": time.time()
        }
        
        self.transactions[payment_id] = transaction
        self.logger.info(f"Payment processed: {payment_id}")
        
        return transaction
    
    def refund_payment(self, payment_result: dict) -> bool:
        """退款"""
        payment_id = payment_result.get("payment_id")
        if payment_id in self.transactions:
            self.transactions[payment_id]["status"] = "refunded"
            self.logger.info(f"Payment refunded: {payment_id}")
            return True
        return False

class MockShippingService:
    """模拟配送服务"""
    
    def __init__(self):
        self.shipments = {}
        self.logger = logging.getLogger("ShippingService")
    
    def schedule_shipping(self, shipping_info: dict) -> dict:
        """安排配送"""
        shipment_id = f"shipment_{uuid.uuid4().hex[:8]}"
        address = shipping_info.get("address", "")
        
        self.logger.info(f"Scheduling shipment to: {address}")
        time.sleep(0.4)  # 模拟配送调度时间
        
        shipment = {
            "shipment_id": shipment_id,
            "address": address,
            "status": "scheduled",
            "estimated_delivery": time.time() + 86400,  # 24小时后
            "created_at": time.time()
        }
        
        self.shipments[shipment_id] = shipment
        self.logger.info(f"Shipping scheduled: {shipment_id}")
        
        return shipment
    
    def cancel_shipping(self, shipment_id: str) -> bool:
        """取消配送"""
        if shipment_id in self.shipments:
            self.shipments[shipment_id]["status"] = "cancelled"
            self.logger.info(f"Shipping cancelled: {shipment_id}")
            return True
        return False

class BankTransferSagaOrchestrator(SagaOrchestrator):
    """银行转账Saga编排器"""
    
    def __init__(self, saga_id: str, source_bank, target_bank, audit_service):
        super().__init__(saga_id)
        self.source_bank = source_bank
        self.target_bank = target_bank
        self.audit_service = audit_service
    
    def transfer_money_saga(self, transfer_data: dict):
        """转账Saga"""
        from_account = transfer_data.get("from_account")
        to_account = transfer_data.get("to_account")
        amount = transfer_data.get("amount")
        
        # 步骤1: 从源账户扣款
        def debit_source_account():
            return self.source_bank.debit_account(from_account, amount)
        
        def credit_source_account():
            return self.source_bank.credit_account(from_account, amount)
        
        step1 = SagaStep(
            step_id="debit_source",
            name="Debit Source Account",
            service_name="source_bank",
            forward_action=debit_source_account,
            compensation_action=credit_source_account
        )
        
        # 步骤2: 向目标账户入账
        def credit_target_account():
            return self.target_bank.credit_account(to_account, amount)
        
        def debit_target_account():
            return self.target_bank.debit_account(to_account, amount)
        
        step2 = SagaStep(
            step_id="credit_target",
            name="Credit Target Account",
            service_name="target_bank",
            forward_action=credit_target_account,
            compensation_action=debit_target_account
        )
        
        # 步骤3: 记录审计日志
        def log_audit():
            return self.audit_service.log_transfer({
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "timestamp": time.time()
            })
        
        def remove_audit_log():
            # 审计日志通常不删除，只是标记为撤销
            return True
        
        step3 = SagaStep(
            step_id="log_audit",
            name="Log Audit Trail",
            service_name="audit_service",
            forward_action=log_audit,
            compensation_action=remove_audit_log
        )
        
        self.add_step(step1)
        self.add_step(step2)
        self.add_step(step3)

class GoStyleSagaOrchestrator:
    """Go风格Saga编排器"""
    
    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps = []
        self.status = SagaStatus.RUNNING
        self.event_channel = asyncio.Queue()
        self.error_channel = asyncio.Queue()
        self.mutex = asyncio.Lock()
        
    async def add_step_async(self, step: SagaStep):
        """异步添加步骤"""
        async with self.mutex:
            self.steps.append(step)
    
    async def execute_async(self) -> bool:
        """异步执行Saga"""
        try:
            # 创建并发执行的任务
            tasks = []
            for step in self.steps:
                task = asyncio.create_task(self._execute_step_async(step))
                tasks.append(task)
            
            # 等待所有步骤完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Step {i} failed: {result}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Saga execution failed: {str(e)}")
            return False
    
    async def _execute_step_async(self, step: SagaStep) -> bool:
        """异步执行步骤"""
        try:
            # 执行正向操作
            if asyncio.iscoroutinefunction(step.forward_action):
                result = await step.forward_action()
            else:
                result = step.forward_action()
            
            step.result = result
            return True
            
        except Exception as e:
            step.error = str(e)
            return False

class SagaPerformanceAnalysis:
    """Saga性能分析"""
    
    @staticmethod
    def analyze_saga_complexity():
        """分析Saga复杂度"""
        complexity = {
            "message_complexity": {
                "forward_execution": "O(n)",  # n个步骤
                "compensation": "O(n)",       # n个步骤（逆序）
                "total_messages": "2n",      # 正向 + 补偿
                "parallel_execution": "possible"  # 可并行执行正向操作
            },
            "time_complexity": {
                "sequential_execution": "O(n*t)",  # n步，每步t时间
                "parallel_execution": "O(max(t))",  # 并行时取最大时间
                "compensation_time": "O(n*t_c)",    # 补偿时间
                "no_blocking": True
            },
            "space_complexity": {
                "orchestrator_memory": "O(n)",  # n个步骤
                "compensation_log": "O(n)",     # 补偿日志
                "state_tracking": "O(n)"        # 状态跟踪
            }
        }
        
        print("=== Saga Performance Analysis ===\n")
        
        for category, metrics in complexity.items():
            print(f"**{category.replace('_', ' ').title()}**")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value}")
            print()
    
    @staticmethod
    def compare_with_2pc_3pc():
        """与2PC/3PC对比"""
        comparison = {
            "Consistency Model": {
                "2PC": "Strong consistency",
                "3PC": "Strong consistency", 
                "Saga": "Eventual consistency"
            },
            "Availability": {
                "2PC": "Low (blocking)",
                "3PC": "Medium",
                "Saga": "High (no blocking)"
            },
            "Performance": {
                "2PC": "Low (2 round trips)",
                "3PC": "Medium (3 round trips)",
                "Saga": "High (no coordination delay)"
            },
            "Fault Tolerance": {
                "2PC": "Poor (coordinator SPOF)",
                "3PC": "Better",
                "Saga": "Excellent (autonomous services)"
            },
            "Complexity": {
                "2PC": "Low",
                "3PC": "Medium",
                "Saga": "High (compensation logic)"
            },
            "Use Cases": {
                "2PC": "Traditional databases",
                "3PC": "Critical systems",
                "Saga": "Microservices, long transactions"
            }
        }
        
        print("=== Saga vs 2PC vs 3PC Comparison ===\n")
        
        for aspect, comparisons in comparison.items():
            print(f"**{aspect}**")
            for approach, value in comparisons.items():
                print(f"  {approach}: {value}")
            print()

class SagaBestPractices:
    """Saga最佳实践"""
    
    @staticmethod
    def design_principles():
        """设计原则"""
        principles = {
            "compensation_design": [
                "Keep compensation operations idempotent",
                "Design compensation to handle partial failures",
                "Make compensation operations commutative",
                "Test compensation paths thoroughly",
                "Consider eventual consistency timeframes"
            ],
            "service_design": [
                "Design services to be independent and autonomous",
                "Implement proper error handling and retries",
                "Use event sourcing for complex state changes",
                "Design APIs to support sagas",
                "Implement proper observability"
            ],
            "orchestration_design": [
                "Keep orchestration logic simple",
                "Implement proper timeout handling",
                "Design for scalability and resilience",
                "Use circuit breakers for external calls",
                "Implement comprehensive monitoring"
            ],
            "data_consistency": [
                "Design database schemas for sagas",
                "Use separate databases per service",
                "Implement eventual consistency patterns",
                "Design for compensating transactions",
                "Consider using event stores"
            ]
        }
        
        print("=== Saga Design Principles ===\n")
        
        for category, guidelines in principles.items():
            print(f"**{category.replace('_', ' ').title()}**")
            for guideline in guidelines:
                print(f"  • {guideline}")
            print()
    
    @staticmethod
    def common_patterns():
        """常见模式"""
        patterns = {
            "choreography_vs_orchestration": {
                "choreography": {
                    "description": "Services react to events from other services",
                    "pros": ["No central coordinator", "Scalable", "Loose coupling"],
                    "cons": ["Complex event flows", "Hard to debug", "Circular dependencies"],
                    "best_for": "Simple sagas with few services"
                },
                "orchestration": {
                    "description": "Central orchestrator coordinates all steps",
                    "pros": ["Centralized control", "Easier debugging", "Clear flow"],
                    "cons": ["Single point of failure", "Coordination overhead", "Tight coupling"],
                    "best_for": "Complex sagas with many services"
                }
            },
            "compensation_strategies": {
                "saga_compensation": {
                    "description": "Each step has a corresponding compensation",
                    "pros": ["Flexible", "Can handle various failure scenarios"],
                    "cons": ["Complex to design", "May not always be possible"]
                },
                "business_compensation": {
                    "description": "Business-level compensating transactions",
                    "pros": ["Simpler logic", "Business-focused"],
                    "cons": ["Less granular", "May miss edge cases"]
                }
            },
            "timeout_handling": {
                "per_step_timeout": {
                    "description": "Each step has its own timeout",
                    "pros": ["Fine-grained control", "Flexible"],
                    "cons": ["Complex configuration", "Hard to tune"]
                },
                "global_timeout": {
                    "description": "Saga has a global timeout",
                    "pros": ["Simple", "Easy to understand"],
                    "cons": ["Less flexible", "May cause premature failures"]
                }
            }
        }
        
        print("=== Common Saga Patterns ===\n")
        
        for pattern_type, options in patterns.items():
            print(f"**{pattern_type.replace('_', ' ').title()}**")
            for option, details in options.items():
                print(f"  **{option.replace('_', ' ').title()}**: {details['description']}")
                if 'pros' in details:
                    print(f"    Pros: {', '.join(details['pros'])}")
                    print(f"    Cons: {', '.join(details['cons'])}")
                    print(f"    Best for: {details['best_for']}")
            print()

# 主演示函数
def demo_saga_pattern():
    """主演示函数"""
    
    print("=== Saga Pattern Complete Demo ===\n")
    
    # 1. 电商订单Saga演示
    print("1. E-Commerce Order Saga Demo")
    print("="*40)
    
    # 创建服务
    order_service = MockOrderService()
    inventory_service = MockInventoryService()
    payment_service = MockPaymentService()
    shipping_service = MockShippingService()
    
    # 创建事件监听器
    event_logger = SagaEventLogger()
    
    # 创建Saga编排器
    saga = ECommerceSagaOrchestrator(
        "ecommerce_saga_001", order_service, inventory_service, 
        payment_service, shipping_service
    )
    
    # 设置事件监听器
    saga.event_listener = event_logger
    
    # 定义订单数据
    order_data = {
        "items": [
            {"item_id": "item1", "quantity": 2},
            {"item_id": "item2", "quantity": 1}
        ],
        "total_amount": 299.99,
        "payment": {
            "amount": 299.99,
            "card_number": "4111111111111111"
        },
        "shipping": {
            "address": "123 Main St, Anytown, USA"
        }
    }
    
    # 创建订单Saga
    saga.create_order_saga(order_data)
    
    # 执行Saga
    print("\nExecuting e-commerce order saga...")
    success = saga.execute()
    
    print(f"\nSaga execution result: {'Success' if success else 'Failed'}")
    
    # 显示状态
    status = saga.get_status()
    print(f"\nSaga final status:")
    print(f"  Status: {status['status']}")
    print(f"  Completed steps: {status['completed_steps']}")
    print(f"  Failed steps: {status['failed_steps']}")
    
    # 显示事件日志
    print(f"\nEvent log summary:")
    for event in event_logger.events[:5]:  # 显示前5个事件
        print(f"  {event.event_type.value} - {event.step_id}")
    
    # 2. 银行转账Saga演示
    print("\n\n2. Bank Transfer Saga Demo")
    print("="*40)
    
    # 创建银行服务（简化版）
    class MockBank:
        def __init__(self, name):
            self.name = name
            self.accounts = {
                "account001": 10000,
                "account002": 5000,
                "account003": 8000
            }
        
        def debit_account(self, account_id: str, amount: float) -> bool:
            if self.accounts.get(account_id, 0) >= amount:
                self.accounts[account_id] -= amount
                print(f"  {self.name}: Debited {amount} from {account_id}")
                return True
            return False
        
        def credit_account(self, account_id: str, amount: float) -> bool:
            self.accounts[account_id] = self.accounts.get(account_id, 0) + amount
            print(f"  {self.name}: Credited {amount} to {account_id}")
            return True
    
    class MockAuditService:
        def log_transfer(self, transfer_info: dict) -> bool:
            print(f"  Audit: Logged transfer {transfer_info['amount']}")
            return True
    
    # 创建服务
    source_bank = MockBank("Bank A")
    target_bank = MockBank("Bank B")
    audit_service = MockAuditService()
    
    # 创建银行转账Saga
    bank_saga = BankTransferSagaOrchestrator(
        "bank_saga_001", source_bank, target_bank, audit_service
    )
    
    transfer_data = {
        "from_account": "account001",
        "to_account": "account002", 
        "amount": 1000
    }
    
    bank_saga.transfer_money_saga(transfer_data)
    
    print("\nExecuting bank transfer saga...")
    success = bank_saga.execute()
    
    print(f"\nBank saga execution result: {'Success' if success else 'Failed'}")
    
    # 3. Saga性能分析
    print("\n\n3. Saga Performance Analysis")
    print("="*40)
    
    performance = SagaPerformanceAnalysis()
    performance.analyze_saga_complexity()
    performance.compare_with_2pc_3pc()
    
    # 4. 最佳实践
    print("\n4. Saga Best Practices")
    print("="*40)
    
    best_practices = SagaBestPractices()
    best_practices.design_principles()
    best_practices.common_patterns()
    
    # 5. 故障场景演示
    print("\n5. Saga Failure Scenario Demo")
    print("="*40)
    
    # 创建带有故障的订单数据
    failing_order_data = {
        "items": [
            {"item_id": "item1", "quantity": 2}
        ],
        "total_amount": 1500.00,  # 大额支付将失败
        "payment": {
            "amount": 1500.00,
            "card_number": "4111111111111111"
        },
        "shipping": {
            "address": "456 Oak St, Somewhere, USA"
        }
    }
    
    failing_saga = ECommerceSagaOrchestrator(
        "failing_saga_001", order_service, inventory_service, 
        payment_service, shipping_service
    )
    
    failing_saga.create_order_saga(failing_order_data)
    
    print("\nExecuting failing order saga (simulates payment failure)...")
    success = failing_saga.execute()
    
    print(f"\nFailing saga result: {'Success' if success else 'Failed (expected)'}")
    
    status = failing_saga.get_status()
    print(f"\nFailing saga status:")
    print(f"  Status: {status['status']}")
    print(f"  Failed steps: {status['failed_steps']}")
    
    print("\n=== Saga Pattern Summary ===")
    print("Saga Pattern provides eventual consistency through:")
    print("• Choreography or Orchestration of local transactions")
    print("• Compensating actions for each successful step")
    print("• No blocking coordination like 2PC/3PC")
    print("• High availability and scalability")
    print("\nKey considerations:")
    print("• Design proper compensation logic")
    print("• Handle eventual consistency timeframes")
    print("• Implement proper error handling and timeouts")
    print("• Choose between choreography and orchestration")
    print("• Design for idempotent operations")

if __name__ == "__main__":
    demo_saga_pattern()
```

这个Saga Pattern学习文档涵盖了：

1. **Saga概述**：Saga模式的核心理念和解决的问题
2. **两种编排模式**：Choreography（编排）和Orchestration（编舞）的详细对比
3. **完整实现**：包含电商订单、银行转账等实际场景的Saga实现
4. **Go语言风格**：使用Python模拟Go的异步并发和通道模式
5. **故障处理**：详细的故障恢复机制和补偿策略
6. **性能分析**：与2PC/3PC的详细对比分析
7. **最佳实践**：设计原则、常见模式和实施建议
8. **应用场景**：电商、银行、微服务等实际应用案例
9. **补偿设计**：如何设计有效的补偿操作
10. **监控和调试**：Saga的观察性和故障诊断

文档提供了从理论到实践的完整Saga学习路径，特别突出了Saga在现代微服务架构中的重要性和实用性。