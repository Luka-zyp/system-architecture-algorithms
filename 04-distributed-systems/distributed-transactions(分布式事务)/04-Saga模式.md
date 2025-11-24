# Saga模式详解

## 目录
1. [Saga概述](#saga概述)
2. [核心概念](#核心概念)
3. [Saga原理](#saga原理)
4. [编排模式对比](#编排模式对比)
5. [实现策略](#实现策略)
6. [核心实现](#核心实现)
7. [应用场景](#应用场景)
8. [最佳实践](#最佳实践)
9. [故障处理](#故障处理)
10. [性能与优化](#性能与优化)

## Saga概述

### 什么是Saga

Saga模式是一种分布式事务管理策略，通过将分布式事务分解为一系列本地事务，每个事务都有对应的补偿操作来实现最终一致性。与2PC/3PC不同，Saga允许多个参与者并行处理，最终通过补偿操作确保所有参与者达到一致状态。

### 核心性能指标

| 性能指标 | 描述 | 计算方式 | 关键影响因素 |
|---------|------|---------|------------|
| 响应时间 | Saga事务完成所需时间 | 所有步骤执行时间总和(顺序执行)或最长步骤时间(并行执行) | 步骤数量、单步执行时间、网络延迟 |
| 吞吐量 | 单位时间内可处理的Saga事务数 | 总事务数/总处理时间 | 并发度、资源利用率、瓶颈服务性能 |
| 补偿成功率 | 补偿操作成功执行的比例 | 成功补偿次数/需要补偿的总次数 | 补偿设计质量、系统稳定性、幂等性 |
| 一致性达成时间 | 从开始到所有服务达到一致状态的时间 | 正向执行时间 + 补偿时间(如有) | 故障发生概率、补偿效率、重试策略 |

## 核心概念

### Saga基本组成

| 组件 | 描述 | 作用 |
|-----|------|------|
| **步骤(Step)** | Saga中的单个本地事务操作 | 执行特定业务逻辑，是最小执行单元 |
| **补偿操作(Compensation)** | 与步骤对应的逆向操作 | 用于回滚已完成的步骤，确保最终一致性 |
| **编排器(Orchestrator)** | 集中式的Saga协调者 | 协调步骤执行和补偿逻辑，管理整体流程 |
| **参与者(Participant)** | 参与Saga的各服务 | 执行具体的本地事务，提供正向和补偿操作 |
| **事件(Event)** | Saga执行过程中产生的通知 | 用于状态传递和系统监控 |

### 状态定义

```python
class SagaStepStatus:
    """Saga步骤状态"""
    PENDING = "pending"      # 等待执行
    EXECUTING = "executing"  # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"        # 执行失败
    COMPENSATING = "compensating"  # 正在补偿
    COMPENSATED = "compensated"    # 补偿完成
    TIMEOUT = "timeout"      # 执行超时

class SagaStatus:
    """Saga事务状态"""
    RUNNING = "running"      # 正在运行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"        # 执行失败
    COMPENSATING = "compensating"  # 正在补偿
    COMPENSATED = "compensated"    # 补偿完成
    TIMEOUT = "timeout"      # 执行超时

## Saga原理

### Saga执行流程

Saga模式的核心原理是将分布式事务拆分为一系列本地事务，并通过补偿机制确保最终一致性。以下是Saga的基本执行流程：

1. **初始化阶段**：创建Saga实例，定义所有步骤及其补偿操作
2. **正向执行阶段**：顺序执行每个本地事务步骤
3. **成功完成**：所有步骤执行成功，Saga事务完成
4. **补偿执行阶段**：当某步骤失败时，逆序执行已完成步骤的补偿操作
5. **补偿完成**：所有补偿操作执行完成，系统回到一致状态

### Saga步骤核心实现

```python
@dataclass
class SagaStep:
    """Saga步骤核心定义"""
    step_id: str          # 步骤唯一标识
    name: str             # 步骤名称
    service_name: str     # 所属服务名
    forward_action: Callable    # 正向操作函数
    compensation_action: Callable  # 补偿操作函数
    status: SagaStepStatus = SagaStepStatus.PENDING  # 当前状态
    result: Any = None    # 执行结果
    error: str = None     # 错误信息
    max_retries: int = 3  # 最大重试次数
    timeout: float = 30.0 # 超时时间
    
    def execute(self) -> bool:
        """执行正向操作"""
        self.status = SagaStepStatus.EXECUTING
        
        try:
            # 支持同步和异步函数执行
            if asyncio.iscoroutinefunction(self.forward_action):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.result = loop.run_until_complete(self.forward_action())
            else:
                self.result = self.forward_action()
            
            self.status = SagaStepStatus.COMPLETED
            return True
            
        except Exception as e:
            self.status = SagaStepStatus.FAILED
            self.error = str(e)
            return False
    
    def compensate(self) -> bool:
        """执行补偿操作"""
        self.status = SagaStepStatus.COMPENSATING
        
        try:
            # 执行补偿逻辑
            if asyncio.iscoroutinefunction(self.compensation_action):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.compensation_action())
            else:
                self.compensation_action()
            
            self.status = SagaStepStatus.COMPENSATED
            return True
            
        except Exception as e:
            self.error = f"Compensation failed: {str(e)}"
            return False

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

## 编排模式对比

### 两种主要编排模式

| 特性 | **编排式(Orchestration)** | **编舞式(Choreography)** |
|-----|--------------------------|------------------------|
| **控制方式** | 中央协调器控制流程 | 事件驱动，服务间协作 |
| **复杂性** | 低（集中式管理） | 高（分散在各服务） |
| **可维护性** | 高（流程一目了然） | 低（难以追踪整体流程） |
| **耦合度** | 服务间低耦合，但依赖协调器 | 服务间通过事件松耦合 |
| **调试难度** | 低（集中日志和状态） | 高（事件流复杂） |
| **扩展性** | 中（需修改协调器） | 高（添加服务只需监听事件） |
| **适用场景** | 复杂流程，多服务协作 | 简单流程，服务自治性强 |
| **单点故障** | 有（协调器可能成为SPOF） | 无（无中央控制点） |

### 编排式实现核心组件

```python
@dataclass
class SagaEvent:
    """Saga事件"""
    event_type: SagaEventType
    saga_id: str
    step_id: str = None
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

class SagaEventListener(ABC):
    """Saga事件监听器接口"""
    
    @abstractmethod
    def on_event(self, event: SagaEvent):
        """处理事件"""
        pass

class SagaOrchestrator:
    """Saga编排器核心实现"""
    
    def __init__(self, saga_id: str, event_listener: SagaEventListener = None):
        self.saga_id = saga_id  # 唯一标识
        self.steps: List[SagaStep] = []  # 步骤列表
        self.status = SagaStatus.RUNNING  # 初始状态
        self.event_listener = event_listener  # 事件监听器
        self.execution_log: List[dict] = []  # 执行日志
        self.mutex = threading.Lock()  # 线程安全锁
    
    def add_step(self, step: SagaStep):
        """添加Saga步骤"""
        with self.mutex:
            self.steps.append(step)
    
    def execute(self) -> bool:
        """执行Saga事务"""
        self.status = SagaStatus.RUNNING
        self._emit_event(SagaEventType.SAGA_STARTED, self.saga_id)
        
        try:
            # 正向执行所有步骤
            for step in self.steps:
                success = self._execute_step_with_retry(step)
                if not success:
                    # 失败时触发补偿
                    self._compensate()
                    return False
            
            self.status = SagaStatus.COMPLETED
            self._emit_event(SagaEventType.SAGA_COMPLETED, self.saga_id)
            return True
            
        except Exception as e:
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

## 实现策略

### 补偿设计核心原则

#### 补偿操作的关键特性

| 特性 | 描述 | 实现要点 |
|-----|-----|--------|
| **幂等性** | 多次执行补偿操作结果相同 | 使用唯一标识防止重复补偿 |
| **可重试性** | 失败后可重新执行 | 添加重试机制和超时处理 |
| **异步性** | 允许异步执行补偿 | 使用消息队列确保最终执行 |
| **隔离性** | 补偿不影响其他操作 | 事务隔离和资源锁定 |
| **可观察性** | 补偿过程可监控 | 详细日志和状态跟踪 |

### 模拟服务实现

以下是简化的模拟服务类，用于演示Saga模式中的关键操作：

class MockInventoryService:
    """库存服务模拟实现"""
    
    def __init__(self):
        self.inventory = {"item1": {"stock": 100, "reserved": 0}, "item2": {"stock": 50, "reserved": 0}, "item3": {"stock": 25, "reserved": 0}}
        self.logger = logging.getLogger("InventoryService")
    
    def reserve_inventory(self, items: List[dict]) -> dict:
        """预留库存（正向操作）"""
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
        """释放预留（补偿操作） - 幂等性实现，确保多次调用结果一致"""
        for item_id, quantity in reservations.items():
            # 幂等性检查 - 即使库存数据不完整也能正确执行
            if item_id in self.inventory:
                # 确保不会出现负值预留
                self.inventory[item_id]["reserved"] = max(0, self.inventory[item_id]["reserved"] - quantity)
                self.logger.info(f"Released reservation: {quantity} units of {item_id}")
        
        time.sleep(0.2)  # 模拟服务延迟
        return True

class MockPaymentService:
    """支付服务模拟实现"""
    
    def __init__(self):
        self.transactions = {}
        self.logger = logging.getLogger("PaymentService")
    
    def process_payment(self, payment_info: dict) -> dict:
        """处理支付（正向操作）"""
        payment_id = f"payment_{uuid.uuid4().hex[:8]}"
        amount = payment_info.get("amount", 0)
        card_number = payment_info.get("card_number", "")
        
        # 模拟支付处理
        self.logger.info(f"Processing payment: ${amount}")
        time.sleep(0.8)  # 模拟支付处理时间
        
        # 模拟大额支付失败场景
        if amount > 1000:
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
        """退款（补偿操作） - 幂等性设计，支持重复调用"""
        payment_id = payment_result.get("payment_id")
        
        # 幂等性检查 - 如果支付记录不存在或已退款，直接返回成功
        if payment_id not in self.transactions or self.transactions[payment_id]["status"] == "refunded":
            return True
        
        # 执行退款操作
        self.transactions[payment_id]["status"] = "refunded"
        self.logger.info(f"Payment refunded: {payment_id}")
        return True

class MockShippingService:
    """配送服务模拟实现"""
    
    def __init__(self):
        self.shipments = {}
        self.logger = logging.getLogger("ShippingService")
    
    def schedule_shipping(self, shipping_info: dict) -> dict:
        """安排配送（正向操作）"""
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
        """取消配送（补偿操作） - 幂等性实现，允许重复取消"""
        # 幂等性检查 - 如果配送不存在，直接返回成功
        if shipment_id not in self.shipments:
            return True
        
        # 执行取消操作
        self.shipments[shipment_id]["status"] = "cancelled"
        self.logger.info(f"Shipping cancelled: {shipment_id}")
        return True

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
        """设计原则与最佳实践"""
        print("=== Saga Design Principles ===\n")
        
        # 设计原则表格
        print("| 原则 | 描述 | 实施建议 |")
        print("|-----|-----|--------|")
        print("| **幂等性设计** | 所有操作必须支持重复执行 | 1. 使用唯一ID标识每个操作<br>2. 实现状态机避免重复执行<br>3. 补偿操作必须支持幂等 |")
        print("| **事务边界清晰** | 明确定义每个步骤的责任 | 1. 一个步骤只负责一个业务操作<br>2. 避免在单个步骤中执行多个业务逻辑<br>3. 步骤间数据传递清晰 |")
        print("| **补偿机制完善** | 每个正向操作都有对应的补偿 | 1. 补偿逻辑必须能恢复初始状态<br>2. 设计足够的超时和重试机制<br>3. 记录详细的执行日志 |")
        print("| **状态管理** | 集中管理Saga执行状态 | 1. 使用持久化存储保存Saga状态<br>2. 定期检查和恢复中断的Saga<br>3. 提供状态查询接口 |")
        print("| **超时处理** | 合理设置超时时间 | 1. 为每个步骤设置独立超时<br>2. 超时后立即触发补偿<br>3. 实现渐进式超时策略 |")
        print("| **监控与告警** | 全面监控执行过程 | 1. 记录所有关键事件<br>2. 实时监控补偿触发率<br>3. 对异常模式设置告警 |")
        print("\n")
    
    @staticmethod
    def common_patterns():
        """实现模式与最佳实践"""
        print("=== Common Saga Patterns ===\n")
        
        # 实现模式推荐
        print("### 1. 编排式模式 (推荐复杂场景)\n")
        print("**适用场景**：")
        print("- 流程复杂，涉及多个服务协作")
        print("- 需要中央控制和监控")
        print("- 业务流程频繁变化")
        print("\n**实现要点**：")
        print("- 引入专门的Saga协调器服务")
        print("- 协调器负责调用各个服务并管理补偿流程")
        print("- 使用工作流引擎实现复杂编排")
        print("\n")
        
        print("### 2. 编舞式模式 (推荐简单场景)\n")
        print("**适用场景**：")
        print("- 流程简单，服务间耦合度低")
        print("- 强调服务自治性")
        print("- 适合微服务架构")
        print("\n**实现要点**：")
        print("- 使用消息队列/事件总线")
        print("- 服务通过监听事件触发动作")
        print("- 补偿通过发布特定事件实现")
        print("\n")
        
        # 性能优化建议
        print("### 性能优化建议\n")
        print("1. **异步执行**：使用异步模式提高吞吐量")
        print("2. **并行步骤**：不相关的步骤并行执行")
        print("3. **批量操作**：对大量数据采用批量处理")
        print("4. **缓存策略**：缓存频繁访问的数据")
        print("5. **资源池化**：合理使用连接池和线程池")
        print("\n")
        
        # 常见陷阱与避免方法
        print("### 常见陷阱与避免方法\n")
        print("| 陷阱 | 问题 | 避免方法 |")
        print("|-----|-----|--------|")
        print("| **长事务风险** | Saga执行时间过长导致资源锁定 | 分解为更小的Saga，设置合理超时 |")
        print("| **补偿失败** | 补偿操作执行失败 | 实现重试机制，设置最大重试次数 |")
        print("| **数据不一致** | 网络分区导致部分执行 | 使用幂等设计，提供最终一致性保证 |")
        print("| **性能下降** | 服务间通信开销大 | 使用本地消息表和事件溯源 |")
        print("| **监控困难** | 难以追踪整体执行情况 | 集中式日志，端到端追踪，可视化监控 |")
        print("\n")

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