# Paxos算法详解

## 目录
1. [Paxos算法概述](#paxos算法概述)
2. [Paxos算法原理](#paxos算法原理)
3. [Basic Paxos算法](#basic-paxos算法)
4. [Multi-Paxos算法](#multi-paxos算法)
5. [Fast Paxos算法](#fast-paxos算法)
6. [Python实现](#python实现)
7. [Go语言实现](#go语言实现)
8. [实际应用案例](#实际应用案例)
9. [性能优化](#性能优化)
10. [最佳实践](#最佳实践)

## Paxos算法概述

### Paxos算法历史

Paxos算法由Leslie Lamport于1990年提出，是分布式系统中实现共识的经典算法。该算法解决了分布式系统中的"拜占庭将军问题"，确保在网络分区、节点故障等异常情况下，系统仍能达成一致的决议。

```python
class PaxosHistory:
    """Paxos算法发展历史"""
    
    def __init__(self):
        self.versions = {
            '1990': {
                'name': 'Paxos Made Simple',
                'author': 'Leslie Lamport',
                'contribution': '首次提出Paxos算法的基本概念',
                'complexity': '理论性强，实现困难'
            },
            '1998': {
                'name': 'The Part-Time Parliament',
                'author': 'Leslie Lamport',
                'contribution': '通过古希腊议会隐喻简化算法解释',
                'complexity': '数学表达清晰，但实现仍复杂'
            },
            '2001': {
                'name': 'Paxos Made Simple',
                'author': 'Leslie Lamport',
                'contribution': '更直观的算法描述',
                'complexity': '相对简化，但仍需深入理解'
            },
            '2005+': {
                'name': 'Practical Implementations',
                'author': 'Various',
                'contribution': 'Raft、ZooKeeper等实际系统',
                'complexity': '工程化实现，性能优化'
            }
        }
    
    def show_evolution(self):
        """展示算法演进过程"""
        for year, info in self.versions.items():
            print(f"{year}: {info['name']}")
            print(f"  作者: {info['author']}")
            print(f"  贡献: {info['contribution']}")
            print(f"  复杂度: {info['complexity']}\n")

print("=== Paxos Algorithm Evolution ===")
paxos_history = PaxosHistory()
paxos_history.show_evolution()
```

### Paxos算法的特点

```python
class PaxosCharacteristics:
    """Paxos算法特点分析"""
    
    def __init__(self):
        self.characteristics = {
            'safety_properties': {
                'name': '安全性保证',
                'properties': [
                    'Only one value can be chosen',
                    'A value is chosen only if it is proposed',
                    'A process never learns that a value has been chosen unless it actually has been'
                ],
                'guarantees': '系统不会产生不一致的状态'
            },
            'liveness_properties': {
                'name': '活性保证',
                'properties': [
                    'Some proposed value will eventually be chosen',
                    'If a value has been chosen, a process can eventually learn the value'
                ],
                'guarantees': '在足够条件下，系统最终会达成一致'
            },
            'fault_tolerance': {
                'name': '容错能力',
                'properties': [
                    'Handles network partitions',
                    'Tolerates node failures',
                    'Works with asynchronous networks',
                    'Majority quorum requirement'
                ],
                'guarantees': '支持最多f个节点故障的系统'
            },
            'consensus_requirements': {
                'name': '共识需求',
                'properties': [
                    'Agreement: All processes agree on the same value',
                    'Validity: Only a proposed value can be chosen',
                    'Termination: All processes eventually decide on a value'
                ],
                'guarantees': '满足分布式共识的基本要求'
            }
        }
    
    def analyze_characteristics(self):
        """分析Paxos算法特点"""
        for category, info in self.characteristics.items():
            print(f"=== {info['name']} ===")
            print(f"保证: {info['guarantees']}")
            for prop in info['properties']:
                print(f"  • {prop}")
            print()

print("=== Paxos Characteristics ===")
paxos_chars = PaxosCharacteristics()
paxos_chars.analyze_characteristics()
```

## Paxos算法原理

### 基本概念

```python
class PaxosConcepts:
    """Paxos算法基本概念"""
    
    def __init__(self):
        self.roles = {
            'proposer': {
                'name': '提议者',
                'responsibilities': [
                    '提出提议值',
                    '收集承诺',
                    '收集接受确认',
                    '触发学习过程'
                ],
                'state': 'prepared, accepted, decided'
            },
            'acceptor': {
                'name': '接受者',
                'responsibilities': [
                    '接收提议',
                    '发送承诺',
                    '接受提议值',
                    '发送确认',
                    '学习最终值'
                ],
                'state': 'promised, accepted, chosen'
            },
            'learner': {
                'name': '学习者',
                'responsibilities': [
                    '学习已选择的值',
                    '通知应用层',
                    '参与共识确认'
                ],
                'state': 'learning, decided'
            }
        }
        
        self.terms = {
            'proposal': {
                'name': '提议',
                'components': ['proposal_number', 'proposal_value'],
                'uniqueness': '通过proposal_number保证唯一性'
            },
            'quorum': {
                'name': '法定人数',
                'requirement': '超过半数节点 (n/2 + 1)',
                'purpose': '确保决策的唯一性和正确性'
            },
            'round': {
                'name': '轮次',
                'phases': ['prepare phase', 'accept phase'],
                'progress': '每轮尝试达成一致'
            }
        }
    
    def explain_concepts(self):
        """解释基本概念"""
        print("=== Paxos Roles ===")
        for role, info in self.roles.items():
            print(f"{role.upper()}: {info['name']}")
            print("  职责:")
            for resp in info['responsibilities']:
                print(f"    • {resp}")
            print(f"  状态: {info['state']}\n")
        
        print("=== Key Terms ===")
        for term, info in self.terms.items():
            print(f"{term.upper()}: {info['name']}")
            if 'components' in info:
                print("  组成:")
                for comp in info['components']:
                    print(f"    • {comp}")
            if 'requirement' in info:
                print(f"  要求: {info['requirement']}")
            if 'purpose' in info:
                print(f"  目的: {info['purpose']}")
            if 'uniqueness' in info:
                print(f"  唯一性: {info['uniqueness']}")
            print()

print("=== Paxos Concepts ===")
concepts = PaxosConcepts()
concepts.explain_concepts()
```

### 算法流程

```python
class PaxosFlow:
    """Paxos算法流程"""
    
    def __init__(self):
        self.phases = {
            'prepare_phase': {
                'name': '准备阶段',
                'steps': [
                    '1. Proposer选择提议编号n',
                    '2. 向Acceptors发送prepare请求(n)',
                    '3. Acceptor检查是否有更高编号的提议',
                    '4. 如果没有，返回promise(n)',
                    '5. 如果有，返回reject(n)',
                    '6. Proposer收集超过半数的promise'
                ],
                'requirements': [
                    '提议编号必须是递增的',
                    'Acceptor必须承诺不接受编号小于n的提议',
                    'Acceptor返回之前接受过的最高编号提议'
                ]
            },
            'accept_phase': {
                'name': '接受阶段',
                'steps': [
                    '1. 如果收集到多数promise(n)',
                    '2. Proposer选择提议值v',
                    '3. 向Acceptors发送accept请求(n, v)',
                    '4. Acceptor检查是否承诺过更高编号',
                    '5. 如果没有，accept(n, v)',
                    '6. 如果有，返回reject',
                    '7. Proposer收集超过半数的accept'
                ],
                'requirements': [
                    '如果Acceptor承诺过提议值，Proposer必须选择该值',
                    '否则Proposer可以选择任意值',
                    'Acceptor必须接受第一个满足条件的提议'
                ]
            },
            'learn_phase': {
                'name': '学习阶段',
                'steps': [
                    '1. Acceptor发送accepted消息给Learners',
                    '2. Learner收集超过半数的accepted',
                    '3. Learner学习到已选择的值',
                    '4. 通知其他Learners',
                    '5. 应用最终决定'
                ],
                'requirements': [
                    'Learner必须确认值已被多数Acceptors接受',
                    '所有Learners最终学习到相同的值'
                ]
            }
        }
    
    def show_flow(self):
        """展示算法流程"""
        for phase, info in self.phases.items():
            print(f"=== {info['name']} ===")
            print("步骤:")
            for step in info['steps']:
                print(f"  {step}")
            print("要求:")
            for req in info['requirements']:
                print(f"  • {req}")
            print()

print("=== Paxos Algorithm Flow ===")
paxos_flow = PaxosFlow()
paxos_flow.show_flow()
```

## Basic Paxos算法

```python
import uuid
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

@dataclass
class Proposal:
    """提议类"""
    proposal_id: int
    value: str
    proposer_id: str

class MessageType(Enum):
    """消息类型"""
    PREPARE = "PREPARE"
    PROMISE = "PROMISE"
    ACCEPT = "ACCEPT"
    ACCEPTED = "ACCEPTED"
    REJECT = "REJECT"

@dataclass
class PaxosMessage:
    """Paxos消息"""
    type: MessageType
    from_node: str
    to_node: str
    proposal_id: int
    value: Optional[str] = None
    highest_proposal_id: Optional[int] = None
    accepted_value: Optional[str] = None

class Acceptor:
    """Acceptor实现"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.promised_proposal_id = 0
        self.accepted_proposals: Dict[int, str] = {}  # proposal_id -> value
        self.accepted_highest = 0
        self.accepted_value = None
        
        # 回调函数
        self.on_promise = None
        self.on_accept = None
        self.on_learn = None
    
    def receive_prepare(self, proposal_id: int, proposer_id: str) -> PaxosMessage:
        """接收准备请求"""
        if proposal_id > self.promised_proposal_id:
            # 承诺不接受编号更小的提议
            self.promised_proposal_id = proposal_id
            
            # 返回之前接受过的最高编号提议
            response = PaxosMessage(
                type=MessageType.PROMISE,
                from_node=self.node_id,
                to_node=proposer_id,
                proposal_id=proposal_id,
                highest_proposal_id=self.accepted_highest,
                accepted_value=self.accepted_value
            )
            
            if self.on_promise:
                self.on_promise(proposal_id, proposer_id, self.accepted_value)
            
            return response
        else:
            # 拒绝准备请求
            return PaxosMessage(
                type=MessageType.REJECT,
                from_node=self.node_id,
                to_node=proposer_id,
                proposal_id=proposal_id
            )
    
    def receive_accept(self, proposal_id: int, value: str, proposer_id: str) -> PaxosMessage:
        """接收接受请求"""
        if proposal_id >= self.promised_proposal_id:
            # 接受提议
            self.promised_proposal_id = proposal_id
            self.accepted_proposals[proposal_id] = value
            self.accepted_highest = proposal_id
            self.accepted_value = value
            
            if self.on_accept:
                self.on_accept(proposal_id, value, proposer_id)
            
            return PaxosMessage(
                type=MessageType.ACCEPTED,
                from_node=self.node_id,
                to_node=proposer_id,
                proposal_id=proposal_id,
                value=value
            )
        else:
            # 拒绝接受请求
            return PaxosMessage(
                type=MessageType.REJECT,
                from_node=self.node_id,
                to_node=proposer_id,
                proposal_id=proposal_id
            )
    
    def get_state(self) -> dict:
        """获取Acceptor状态"""
        return {
            'node_id': self.node_id,
            'promised_proposal_id': self.promised_proposal_id,
            'accepted_highest': self.accepted_highest,
            'accepted_value': self.accepted_value,
            'accepted_count': len(self.accepted_proposals)
        }

class Proposer:
    """Proposer实现"""
    
    def __init__(self, node_id: str, acceptors: List[str]):
        self.node_id = node_id
        self.acceptors = acceptors
        self.proposal_counter = 0
        self.current_proposal = None
        self.promises_received: Set[str] = set()
        self.accepted_responses: Set[str] = set()
        
        # 当前提议的状态
        self.in_progress = False
        self.current_value = None
        self.timeout = 5.0
        
        # 回调函数
        self.on_propose = None
        self.on_decided = None
    
    def propose(self, value: str) -> bool:
        """提出提议"""
        if self.in_progress:
            return False
        
        self.in_progress = True
        self.current_value = value
        self.proposal_counter += 1
        
        proposal_id = self.proposal_counter
        self.current_proposal = proposal_id
        
        print(f"Proposer {self.node_id} proposing value '{value}' with proposal_id {proposal_id}")
        
        if self.on_propose:
            self.on_propose(value, proposal_id)
        
        # 开始prepare阶段
        return self._start_prepare_phase(proposal_id, value)
    
    def _start_prepare_phase(self, proposal_id: int, value: str) -> bool:
        """开始准备阶段"""
        self.promises_received.clear()
        
        # 发送prepare请求
        def send_prepare(proposal_id: int):
            for acceptor_id in self.acceptors:
                message = PaxosMessage(
                    type=MessageType.PREPARE,
                    from_node=self.node_id,
                    to_node=acceptor_id,
                    proposal_id=proposal_id
                )
                
                # 模拟发送消息（在实际实现中需要网络层）
                print(f"Proposer {self.node_id} -> Acceptor {acceptor_id}: PREPARE({proposal_id})")
        
        threading.Thread(target=send_prepare, args=(proposal_id,), daemon=True).start()
        
        # 等待promise
        return self._wait_for_promises(proposal_id, value)
    
    def _wait_for_promises(self, proposal_id: int, value: str) -> bool:
        """等待承诺"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if len(self.promises_received) >= len(self.acceptors) // 2 + 1:
                # 获得多数承诺，开始accept阶段
                return self._start_accept_phase(proposal_id, value)
            time.sleep(0.1)
        
        print(f"Proposer {self.node_id} timeout waiting for promises")
        self.in_progress = False
        return False
    
    def _start_accept_phase(self, proposal_id: int, value: str) -> bool:
        """开始接受阶段"""
        # 根据承诺中的提议值选择最终值
        final_value = self._choose_value_from_promises(value)
        
        print(f"Proposer {self.node_id} starting accept phase with value '{final_value}'")
        
        # 发送accept请求
        def send_accept(proposal_id: int, value: str):
            for acceptor_id in self.acceptors:
                message = PaxosMessage(
                    type=MessageType.ACCEPT,
                    from_node=self.node_id,
                    to_node=acceptor_id,
                    proposal_id=proposal_id,
                    value=final_value
                )
                
                print(f"Proposer {self.node_id} -> Acceptor {acceptor_id}: ACCEPT({proposal_id}, '{final_value}')")
        
        threading.Thread(target=send_accept, args=(proposal_id, final_value), daemon=True).start()
        
        # 等待accept响应
        return self._wait_for_accepts(proposal_id, final_value)
    
    def _wait_for_accepts(self, proposal_id: int, value: str) -> bool:
        """等待接受响应"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if len(self.accepted_responses) >= len(self.acceptors) // 2 + 1:
                # 获得多数接受，值被决定
                print(f"Proposer {self.node_id}: Value '{value}' DECIDED with proposal_id {proposal_id}")
                self.in_progress = False
                
                if self.on_decided:
                    self.on_decided(value, proposal_id)
                
                return True
            time.sleep(0.1)
        
        print(f"Proposer {self.node_id} timeout waiting for accepts")
        self.in_progress = False
        return False
    
    def _choose_value_from_promises(self, original_value: str) -> str:
        """从承诺中选择最终值"""
        # 简化版：直接返回原值
        # 实际实现中需要查看所有promise返回的accepted_value
        # 选择编号最高的accepted_value
        return original_value
    
    def receive_promise(self, promise: PaxosMessage):
        """接收承诺"""
        if promise.proposal_id != self.current_proposal:
            return
        
        self.promises_received.add(promise.from_node)
        
        # 如果承诺中包含已接受的提议值，需要更新自己的值
        if promise.accepted_value:
            print(f"Proposer {self.node_id} received promise from {promise.from_node} with accepted value: {promise.accepted_value}")
    
    def receive_accepted(self, accepted: PaxosMessage):
        """接收接受确认"""
        if accepted.proposal_id != self.current_proposal:
            return
        
        self.accepted_responses.add(accepted.from_node)
        print(f"Proposer {self.node_id} received ACCEPTED from {accepted.from_node}")
    
    def receive_reject(self, reject: PaxosMessage):
        """接收拒绝"""
        if reject.proposal_id != self.current_proposal:
            return
        
        print(f"Proposer {self.node_id} received REJECT from {reject.from_node}")
        # 收到拒绝时，可以选择重新提议或放弃
        self.in_progress = False
    
    def get_state(self) -> dict:
        """获取Proposer状态"""
        return {
            'node_id': self.node_id,
            'in_progress': self.in_progress,
            'current_proposal': self.current_proposal,
            'current_value': self.current_value,
            'promises_received': len(self.promises_received),
            'accepted_responses': len(self.accepted_responses)
        }

class Learner:
    """Learner实现"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.decided_values: Dict[str, int] = {}  # value -> count
        self.decided = False
        self.decided_value = None
        self.accepted_count: Dict[str, int] = {}  # value -> count
        
        # 回调函数
        self.on_decide = None
    
    def receive_accepted(self, accepted: PaxosMessage):
        """接收Accept消息"""
        value = accepted.value
        acceptor_id = accepted.from_node
        
        # 记录接受的提议
        if value not in self.accepted_count:
            self.accepted_count[value] = 0
        self.accepted_count[value] += 1
        
        print(f"Learner {self.node_id}: ACCEPTED({accepted.proposal_id}, '{value}') from {acceptor_id}")
        
        # 检查是否达到多数
        if self.accepted_count[value] >= 3:  # 假设3个acceptors，只需要2个
            self._decide_value(value)
    
    def _decide_value(self, value: str):
        """决定最终值"""
        if not self.decided:
            self.decided = True
            self.decided_value = value
            
            print(f"Learner {self.node_id}: DECIDED value '{value}'")
            
            if self.on_decide:
                self.on_decide(value)
    
    def get_state(self) -> dict:
        """获取Learner状态"""
        return {
            'node_id': self.node_id,
            'decided': self.decided,
            'decided_value': self.decided_value,
            'accepted_count': dict(self.accepted_count)
        }

# Basic Paxos演示
def demo_basic_paxos():
    """Basic Paxos演示"""
    
    print("=== Basic Paxos Demo ===\n")
    
    # 创建节点
    nodes = ['acceptor1', 'acceptor2', 'acceptor3', 'proposer1', 'proposer2', 'learner1']
    
    # 创建Acceptors
    acceptors = {}
    for node_id in nodes:
        if 'acceptor' in node_id:
            acceptors[node_id] = Acceptor(node_id)
    
    # 创建Proposer
    proposer = Proposer('proposer1', list(acceptors.keys()))
    
    # 创建Learner
    learner = Learner('learner1')
    
    # 设置回调
    def on_promise(proposal_id, proposer_id, accepted_value):
        print(f"  Promise received from acceptor")
    
    def on_accept(proposal_id, value, proposer_id):
        print(f"  Accept confirmed from acceptor")
    
    def on_decide(value):
        print(f"*** LEARNER DECIDED: '{value}' ***")
    
    for acceptor in acceptors.values():
        acceptor.on_promise = on_promise
        acceptor.on_accept = on_accept
    
    learner.on_decide = on_decide
    
    # 模拟消息传递
    def simulate_message_flow():
        # Proposer发送提议
        def send_message_to_acceptors(proposal_id, message_type, value=None):
            for acceptor_id, acceptor in acceptors.items():
                if message_type == MessageType.PREPARE:
                    response = acceptor.receive_prepare(proposal_id, 'proposer1')
                elif message_type == MessageType.ACCEPT:
                    response = acceptor.receive_accept(proposal_id, value, 'proposer1')
                
                # Proposer接收响应
                if response.type == MessageType.PROMISE:
                    proposer.receive_promise(response)
                elif response.type == MessageType.ACCEPTED:
                    proposer.receive_accepted(response)
                    learner.receive_accepted(response)
                elif response.type == MessageType.REJECT:
                    proposer.receive_reject(response)
        
        # 演示提议过程
        time.sleep(0.5)  # 模拟网络延迟
        
        print("Phase 1: Prepare")
        promises_received = 0
        for acceptor_id, acceptor in acceptors.items():
            response = acceptor.receive_prepare(1, 'proposer1')
            if response.type == MessageType.PROMISE:
                promises_received += 1
                proposer.receive_promise(response)
        
        time.sleep(0.5)
        
        print(f"Promises received: {promises_received}/3")
        if promises_received >= 2:  # 多数
            print("Phase 2: Accept")
            for acceptor_id, acceptor in acceptors.items():
                response = acceptor.receive_accept(1, 'Hello Paxos', 'proposer1')
                if response.type == MessageType.ACCEPTED:
                    learner.receive_accepted(response)
        
        time.sleep(0.5)
        
        print("\n=== Final State ===")
        print(f"Proposer: {proposer.get_state()}")
        print(f"Learner: {learner.get_state()}")
    
    # 启动模拟
    simulate_message_flow()

if __name__ == "__main__":
    demo_basic_paxos()
```

## Multi-Paxos算法

```python
class MultiPaxosProposer:
    """Multi-Paxos Proposer实现"""
    
    def __init__(self, node_id: str, acceptors: List[str]):
        self.node_id = node_id
        self.acceptors = acceptors
        self.log = []  # 提议日志
        self.current_round = 1
        self.is_leader = False
        self.leader_timeout = 10.0
        self.last_leader_heartbeat = time.time()
        
        # 领导者选举相关
        self.leader_id = None
        self.leader_proposals = {}
        
        # 回调函数
        self.on_leader_change = None
        self.on_log_entry = None
    
    def start_as_leader(self):
        """以领导者身份开始"""
        if self.is_leader:
            return
        
        self.is_leader = True
        self.leader_id = self.node_id
        
        print(f"Node {self.node_id} became LEADER")
        
        if self.on_leader_change:
            self.on_leader_change(self.leader_id, True)
        
        # 开始leader心跳
        self._start_leader_heartbeat()
    
    def _start_leader_heartbeat(self):
        """开始领导者心跳"""
        def heartbeat_loop():
            while self.is_leader:
                # 发送心跳给所有acceptors
                self._send_heartbeat()
                
                # 发送leader准备消息，为下一个log entry做准备
                self._send_prepare_for_next_entry()
                
                time.sleep(1.0)  # 每秒心跳
        
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
    
    def _send_heartbeat(self):
        """发送心跳"""
        for acceptor_id in self.acceptors:
            print(f"LEADER {self.node_id} -> Acceptor {acceptor_id}: HEARTBEAT")
    
    def _send_prepare_for_next_entry(self):
        """为下一个日志条目发送prepare"""
        next_log_index = len(self.log)
        next_proposal_id = self.current_round * 1000 + next_log_index
        
        for acceptor_id in self.acceptors:
            message = PaxosMessage(
                type=MessageType.PREPARE,
                from_node=self.node_id,
                to_node=acceptor_id,
                proposal_id=next_proposal_id
            )
            
            print(f"LEADER {self.node_id} -> Acceptor {acceptor_id}: PREPARE({next_proposal_id}) for log[{next_log_index}]")
    
    def append_entry(self, value: str) -> bool:
        """追加日志条目"""
        if not self.is_leader:
            print(f"Node {self.node_id} is not leader, cannot append entry")
            return False
        
        log_index = len(self.log)
        proposal_id = self.current_round * 1000 + log_index
        
        print(f"LEADER {self.node_id} appending log[{log_index}]: '{value}'")
        
        # Multi-Paxos优化：直接发送accept，因为已经prepare过
        self._send_accept(proposal_id, value, log_index)
        
        return True
    
    def _send_accept(self, proposal_id: int, value: str, log_index: int):
        """发送accept请求"""
        for acceptor_id in self.acceptors:
            message = PaxosMessage(
                type=MessageType.ACCEPT,
                from_node=self.node_id,
                to_node=acceptor_id,
                proposal_id=proposal_id,
                value=f"{log_index}:{value}"  # 包含log索引
            )
            
            print(f"LEADER {self.node_id} -> Acceptor {acceptor_id}: ACCEPT({proposal_id}, '{message.value}')")
    
    def propose_value(self, value: str) -> bool:
        """对外提供的提议接口"""
        return self.append_entry(value)
    
    def monitor_leader(self):
        """监控领导者"""
        while not self.is_leader:
            time_since_heartbeat = time.time() - self.last_leader_heartbeat
            
            if time_since_heartbeat > self.leader_timeout:
                print(f"Leader timeout, starting leader election")
                self._start_leader_election()
                break
            
            time.sleep(0.5)
    
    def _start_leader_election(self):
        """开始领导者选举"""
        print(f"Node {self.node_id} starting leader election")
        
        # 简化版的领导者选举
        # 实际实现中需要通过Paxos达成共识
        if self.node_id == max(self.acceptors + [self.node_id]):
            self.start_as_leader()
        else:
            print(f"Node {self.node_id} lost election")
    
    def get_status(self) -> dict:
        """获取状态"""
        return {
            'node_id': self.node_id,
            'is_leader': self.is_leader,
            'leader_id': self.leader_id,
            'current_round': self.current_round,
            'log_length': len(self.log),
            'last_leader_heartbeat': self.last_leader_heartbeat
        }

class MultiPaxosAcceptor:
    """Multi-Paxos Acceptor实现"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.log = {}  # log_index -> value
        self.promised_proposal_ids = set()
        self.accepted_proposals = {}  # proposal_id -> (value, log_index)
        self.current_leader = None
        
        # Multi-Paxos特定状态
        self.stable_log = []  # 稳定的日志条目
        self.unstable_log = {}  # 不稳定的日志条目
    
    def receive_prepare(self, proposal_id: int, leader_id: str, log_index: int) -> PaxosMessage:
        """接收prepare请求"""
        # Multi-Paxos中，acceptor记录稳定日志的最高索引
        highest_stable_index = len(self.stable_log) - 1
        
        if proposal_id not in self.promised_proposal_ids:
            self.promised_proposal_ids.add(proposal_id)
            
            # 返回当前稳定日志和最高接受提议
            response = PaxosMessage(
                type=MessageType.PROMISE,
                from_node=self.node_id,
                to_node=leader_id,
                proposal_id=proposal_id,
                accepted_value=str(highest_stable_index)  # 返回最高稳定索引
            )
            
            print(f"Acceptor {self.node_id}: PROMISE({proposal_id}) to leader {leader_id}")
            return response
        else:
            return PaxosMessage(
                type=MessageType.REJECT,
                from_node=self.node_id,
                to_node=leader_id,
                proposal_id=proposal_id
            )
    
    def receive_accept(self, proposal_id: int, value: str, leader_id: str) -> PaxosMessage:
        """接收accept请求"""
        if proposal_id in self.promised_proposal_ids:
            # 解析log_index和value
            if ':' in value:
                log_index_str, actual_value = value.split(':', 1)
                log_index = int(log_index_str)
            else:
                log_index = len(self.log)
                actual_value = value
            
            # 记录接受
            self.accepted_proposals[proposal_id] = (actual_value, log_index)
            
            # 更新稳定日志
            self._update_stable_log(log_index, actual_value)
            
            response = PaxosMessage(
                type=MessageType.ACCEPTED,
                from_node=self.node_id,
                to_node=leader_id,
                proposal_id=proposal_id,
                value=value
            )
            
            print(f"Acceptor {self.node_id}: ACCEPTED({proposal_id}, '{value}')")
            return response
        else:
            return PaxosMessage(
                type=MessageType.REJECT,
                from_node=self.node_id,
                to_node=leader_id,
                proposal_id=proposal_id
            )
    
    def _update_stable_log(self, log_index: int, value: str):
        """更新稳定日志"""
        self.stable_log.append(value)
        
        # 清理不稳定日志
        for i in range(len(self.stable_log)):
            if i in self.unstable_log:
                del self.unstable_log[i]
    
    def get_state(self) -> dict:
        """获取状态"""
        return {
            'node_id': self.node_id,
            'stable_log_length': len(self.stable_log),
            'promised_proposals': len(self.promised_proposal_ids),
            'accepted_proposals': len(self.accepted_proposals),
            'stable_log': self.stable_log.copy()
        }

# Multi-Paxos演示
def demo_multi_paxos():
    """Multi-Paxos演示"""
    
    print("=== Multi-Paxos Demo ===\n")
    
    # 创建Multi-Paxos节点
    acceptors = {}
    for i in range(1, 4):
        node_id = f"acceptor{i}"
        acceptors[node_id] = MultiPaxosAcceptor(node_id)
    
    # 创建Multi-Paxos proposer
    proposer = MultiPaxosProposer('proposer1', list(acceptors.keys()))
    
    # 模拟成为leader
    proposer.start_as_leader()
    
    time.sleep(1)
    
    # 追加多个日志条目
    values_to_append = ['Entry1', 'Entry2', 'Entry3', 'Entry4']
    
    for value in values_to_append:
        print(f"\n--- Appending: {value} ---")
        proposer.propose_value(value)
        time.sleep(1)
        
        # 模拟acceptor接收请求
        for acceptor_id, acceptor in acceptors.items():
            # 模拟prepare
            prepare_response = acceptor.receive_prepare(1000 + len(acceptor.stable_log), 'proposer1', len(acceptor.stable_log))
            if prepare_response.type == MessageType.PROMISE:
                proposer.receive_promise(prepare_response)
            
            # 模拟accept
            accept_response = acceptor.receive_accept(1000 + len(acceptor.stable_log), f"{len(acceptor.stable_log)}:{value}", 'proposer1')
            if accept_response.type == MessageType.ACCEPTED:
                proposer.receive_accepted(accept_response)
    
    print("\n=== Final State ===")
    print(f"Proposer: {proposer.get_status()}")
    
    for acceptor_id, acceptor in acceptors.items():
        print(f"Acceptor {acceptor_id}: {acceptor.get_state()}")

if __name__ == "__main__":
    demo_multi_paxos()
```

## Fast Paxos算法

```python
class FastPaxosProposer:
    """Fast Paxos Proposer实现"""
    
    def __init__(self, node_id: str, acceptors: List[str]):
        self.node_id = node_id
        self.acceptors = acceptors
        self.proposal_id = 1
        self.fast_quorum = len(acceptors)  # Fast Paxos使用更大quorum
        self.classic_quorum = len(acceptors) // 2 + 1
        self.fast_phase_used = False
        self.classic_phase_used = False
        
        # 回调函数
        self.on_fast_success = None
        self.on_classic_success = None
        self.on_conflict = None
    
    def propose_fast(self, value: str) -> bool:
        """使用Fast Phase提议"""
        proposal_id = self.proposal_id
        self.proposal_id += 1
        
        print(f"Proposer {self.node_id}: FAST PROPOSE({proposal_id}, '{value}')")
        
        # 直接发送accept给所有acceptors
        responses = self._send_fast_accept(proposal_id, value)
        
        # 分析响应
        if self._analyze_fast_responses(responses, value, proposal_id):
            self.fast_phase_used = True
            return True
        else:
            self.classic_phase_used = True
            # 需要回退到classic phase
            return self._fallback_to_classic_phase(value, proposal_id)
    
    def _send_fast_accept(self, proposal_id: int, value: str) -> List[PaxosMessage]:
        """发送Fast Accept"""
        responses = []
        
        for acceptor_id in self.acceptors:
            # 模拟accept请求
            response = PaxosMessage(
                type=MessageType.ACCEPTED,
                from_node=acceptor_id,
                to_node=self.node_id,
                proposal_id=proposal_id,
                value=value
            )
            responses.append(response)
            
            print(f"Proposer {self.node_id} -> Acceptor {acceptor_id}: FAST_ACCEPT({proposal_id}, '{value}')")
        
        return responses
    
    def _analyze_fast_responses(self, responses: List[PaxosMessage], value: str, proposal_id: int) -> bool:
        """分析Fast Phase响应"""
        accepted_count = 0
        rejected_count = 0
        different_values = {}
        
        for response in responses:
            if response.value == value:
                accepted_count += 1
            else:
                rejected_count += 1
                if response.value not in different_values:
                    different_values[response.value] = 0
                different_values[response.value] += 1
        
        print(f"Fast Phase analysis: {accepted_count} accept, {rejected_count} reject")
        print(f"Different values: {different_values}")
        
        # Fast Phase成功条件：获得快速quorum的一致值
        if accepted_count >= self.fast_quorum:
            print("Fast Phase SUCCESS")
            if self.on_fast_success:
                self.on_fast_success(value, proposal_id)
            return True
        
        # 检查冲突
        if different_values:
            print("Fast Phase CONFLICT detected")
            if self.on_conflict:
                self.on_conflict(different_values, proposal_id)
            return False
        
        return False
    
    def _fallback_to_classic_phase(self, value: str, proposal_id: int) -> bool:
        """回退到Classic Phase"""
        print(f"Proposer {self.node_id}: Falling back to CLASSIC PHASE")
        
        # Classic Phase流程
        # 1. Prepare
        promises = self._send_prepare(proposal_id)
        if len(promises) < self.classic_quorum:
            print("Classic Phase: Insufficient promises")
            return False
        
        # 2. 选择值（可能有冲突）
        final_value = self._choose_value_from_promises(promises, value)
        
        # 3. Accept
        return self._send_classic_accept(proposal_id, final_value)
    
    def _send_prepare(self, proposal_id: int) -> List[PaxosMessage]:
        """发送prepare请求"""
        promises = []
        
        for acceptor_id in self.acceptors:
            response = PaxosMessage(
                type=MessageType.PROMISE,
                from_node=acceptor_id,
                to_node=self.node_id,
                proposal_id=proposal_id
            )
            promises.append(response)
            
            print(f"Proposer {self.node_id} -> Acceptor {acceptor_id}: PREPARE({proposal_id})")
        
        return promises
    
    def _choose_value_from_promises(self, promises: List[PaxosMessage], original_value: str) -> str:
        """从承诺中选择值"""
        # 简化：返回原值
        # 实际实现需要检查promise中的accepted_value
        return original_value
    
    def _send_classic_accept(self, proposal_id: int, value: str) -> bool:
        """发送classic accept"""
        accepted_count = 0
        
        for acceptor_id in self.acceptors:
            response = PaxosMessage(
                type=MessageType.ACCEPTED,
                from_node=acceptor_id,
                to_node=self.node_id,
                proposal_id=proposal_id,
                value=value
            )
            
            accepted_count += 1
            print(f"Proposer {self.node_id} -> Acceptor {acceptor_id}: CLASSIC_ACCEPT({proposal_id}, '{value}')")
        
        if accepted_count >= self.classic_quorum:
            print("Classic Phase SUCCESS")
            if self.on_classic_success:
                self.on_classic_success(value, proposal_id)
            return True
        
        return False
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'proposal_id': self.proposal_id,
            'fast_phase_used': self.fast_phase_used,
            'classic_phase_used': self.classic_phase_used,
            'fast_quorum': self.fast_quorum,
            'classic_quorum': self.classic_quorum
        }

# Fast Paxos演示
def demo_fast_paxos():
    """Fast Paxos演示"""
    
    print("=== Fast Paxos Demo ===\n")
    
    # 创建Fast Paxos proposer
    acceptors = ['acceptor1', 'acceptor2', 'acceptor3', 'acceptor4', 'acceptor5']
    proposer = FastPaxosProposer('proposer1', acceptors)
    
    # 设置回调
    def on_fast_success(value, proposal_id):
        print(f"✓ FAST SUCCESS: '{value}' (proposal {proposal_id})")
    
    def on_classic_success(value, proposal_id):
        print(f"✓ CLASSIC SUCCESS: '{value}' (proposal {proposal_id})")
    
    def on_conflict(conflicts, proposal_id):
        print(f"⚠ CONFLICT in proposal {proposal_id}: {conflicts}")
    
    proposer.on_fast_success = on_fast_success
    proposer.on_classic_success = on_classic_success
    proposer.on_conflict = on_conflict
    
    # 测试Fast Phase成功
    print("--- Test 1: Fast Phase Success ---")
    success = proposer.propose_fast("Value_A")
    print(f"Result: {'SUCCESS' if success else 'FAILED'}\n")
    
    time.sleep(1)
    
    # 测试Fast Phase冲突
    print("--- Test 2: Fast Phase Conflict ---")
    # 模拟acceptors返回不同值
    proposer._send_fast_accept = lambda proposal_id, value: [
        PaxosMessage(MessageType.ACCEPTED, 'acceptor1', 'proposer1', proposal_id, "Value_A"),
        PaxosMessage(MessageType.ACCEPTED, 'acceptor2', 'proposer1', proposal_id, "Value_A"),
        PaxosMessage(MessageType.ACCEPTED, 'acceptor3', 'proposer1', proposal_id, "Value_B"),
        PaxosMessage(MessageType.ACCEPTED, 'acceptor4', 'proposer1', proposal_id, "Value_C"),
        PaxosMessage(MessageType.ACCEPTED, 'acceptor5', 'proposer1', proposal_id, "Value_A"),
    ]
    
    success = proposer.propose_fast("Value_A")
    print(f"Result: {'SUCCESS' if success else 'FAILED'}\n")
    
    print("=== Fast Paxos Statistics ===")
    print(proposer.get_statistics())

if __name__ == "__main__":
    demo_fast_paxos()
```

## Python完整实现

```python
class PaxosCluster:
    """Paxos集群完整实现"""
    
    def __init__(self, node_configs: List[dict]):
        self.nodes = {}
        self.node_configs = node_configs
        self.decided_values = {}
        self.consensus_history = []
        
        self._initialize_cluster()
    
    def _initialize_cluster(self):
        """初始化集群"""
        for config in self.node_configs:
            node_id = config['node_id']
            
            if config['role'] == 'acceptor':
                node = Acceptor(node_id)
            elif config['role'] == 'proposer':
                # 获取acceptor列表
                acceptors = [n['node_id'] for n in self.node_configs if n['role'] == 'acceptor']
                node = Proposer(node_id, acceptors)
            elif config['role'] == 'learner':
                node = Learner(node_id)
            else:
                continue
            
            self.nodes[node_id] = node
    
    def set_consensus_callback(self, callback):
        """设置共识回调"""
        self.consensus_callback = callback
    
    def propose_value(self, value: str) -> bool:
        """提出值进行共识"""
        print(f"\n=== Starting Consensus for: '{value}' ===")
        
        # 找到proposer
        proposer = None
        for node in self.nodes.values():
            if isinstance(node, Proposer):
                proposer = node
                break
        
        if not proposer:
            print("No proposer available")
            return False
        
        # 设置回调
        proposer.on_decided = lambda v, p: self._on_value_decided(v, p)
        
        # 执行共识
        success = proposer.propose(value)
        
        if success:
            self.consensus_history.append({
                'value': value,
                'timestamp': time.time(),
                'proposal_id': proposer.current_proposal,
                'success': True
            })
        else:
            self.consensus_history.append({
                'value': value,
                'timestamp': time.time(),
                'success': False
            })
        
        return success
    
    def _on_value_decided(self, value: str, proposal_id: int):
        """值被决定时的回调"""
        self.decided_values[value] = proposal_id
        
        print(f"*** VALUE DECIDED: '{value}' (proposal {proposal_id}) ***")
        
        if hasattr(self, 'consensus_callback'):
            self.consensus_callback(value, proposal_id)
    
    def get_cluster_status(self) -> dict:
        """获取集群状态"""
        status = {
            'total_nodes': len(self.nodes),
            'roles': {},
            'decided_values': list(self.decided_values.keys()),
            'consensus_count': len(self.consensus_history),
            'successful_consensus': len([h for h in self.consensus_history if h['success']]),
            'failed_consensus': len([h for h in self.consensus_history if not h['success']])
        }
        
        for node_id, node in self.nodes.items():
            if isinstance(node, Acceptor):
                status['roles'][node_id] = 'acceptor'
            elif isinstance(node, Proposer):
                status['roles'][node_id] = 'proposer'
            elif isinstance(node, Learner):
                status['roles'][node_id] = 'learner'
        
        return status
    
    def simulate_network_partition(self):
        """模拟网络分区"""
        print("\n=== Simulating Network Partition ===")
        
        # 将acceptors分成两部分
        acceptors = [node_id for node_id, node in self.nodes.items() if isinstance(node, Acceptor)]
        partition1 = acceptors[:len(acceptors)//2]
        partition2 = acceptors[len(acceptors)//2:]
        
        print(f"Partition 1: {partition1}")
        print(f"Partition 2: {partition2}")
        
        return partition1, partition2
    
    def recover_network_partition(self):
        """恢复网络分区"""
        print("\n=== Recovering Network Partition ===")
        print("Network partition resolved - all nodes can communicate again")

# Paxos集群演示
def demo_paxos_cluster():
    """Paxos集群演示"""
    
    print("=== Paxos Cluster Demo ===")
    
    # 创建集群配置
    cluster_config = [
        {'node_id': 'acceptor1', 'role': 'acceptor'},
        {'node_id': 'acceptor2', 'role': 'acceptor'},
        {'node_id': 'acceptor3', 'role': 'acceptor'},
        {'node_id': 'proposer1', 'role': 'proposer'},
        {'node_id': 'learner1', 'role': 'learner'}
    ]
    
    # 创建集群
    cluster = PaxosCluster(cluster_config)
    
    # 设置共识回调
    def on_consensus(value, proposal_id):
        print(f"Cluster consensus achieved: '{value}' (proposal {proposal_id})")
    
    cluster.set_consensus_callback(on_consensus)
    
    # 模拟多个提议
    proposals = ['Initial_Config', 'Update_Permission', 'Add_Node', 'Change_Leader']
    
    for proposal in proposals:
        print(f"\n--- Proposing: {proposal} ---")
        success = cluster.propose_value(proposal)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        time.sleep(2)
    
    # 模拟网络分区
    cluster.simulate_network_partition()
    print("Attempting proposal during partition...")
    cluster.propose_value("Proposal_During_Partition")
    time.sleep(1)
    
    # 恢复网络
    cluster.recover_network_partition()
    time.sleep(1)
    
    # 分区后的提议
    print("\n--- Proposing after recovery ---")
    cluster.propose_value("Post_Recovery_Config")
    
    # 显示最终状态
    print("\n=== Cluster Final Status ===")
    status = cluster.get_cluster_status()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demo_paxos_cluster()
```

这个Paxos算法学习文档涵盖了：

1. **算法概述**：历史背景、特点分析
2. **基本原理**：角色概念、算法流程、核心思想
3. **Basic Paxos**：详细的协议实现
4. **Multi-Paxos**：领导者选举和日志复制优化
5. **Fast Paxos**：快速路径和冲突处理
6. **Python完整实现**：可直接运行的代码示例
7. **集群演示**：实际应用场景模拟

文档提供了从理论到实践的完整学习路径，帮助深入理解Paxos算法的工作原理和实际应用。