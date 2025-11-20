# Kubernetes安全最佳实践 (Kubernetes Security Best Practices)

## 目录
1. [Kubernetes安全概述](#kubernetes安全概述)
2. [安全威胁分析](#安全威胁分析)
3. [多层安全架构](#多层安全架构)
4. [身份认证与授权](#身份认证与授权)
5. [网络安全策略](#网络安全策略)
6. [Pod与容器安全](#pod与容器安全)
7. [镜像安全管理](#镜像安全管理)
8. [数据加密与密钥管理](#数据加密与密钥管理)
9. [安全监控与审计](#安全监控与审计)
10. [合规性管理](#合规性管理)
11. [实施场景与案例](#实施场景与案例)
12. [最佳实践总结](#最佳实践总结)

## Kubernetes安全概述

### 什么是Kubernetes安全？

Kubernetes安全是一个综合性的安全框架，旨在保护容器化应用程序在Kubernetes集群中的安全。它涉及多个层次的安全控制，从集群级别的访问控制到单个容器的运行时安全。

### 核心安全原则

```mermaid
graph TB
    A[最小权限原则] --> B[零信任架构]
    C[深度防御] --> B
    D[纵深防御] --> B
    E[安全左移] --> B
    F[持续监控] --> B
    
    G[身份认证] --> H[授权控制]
    I[网络安全] --> H
    J[数据加密] --> H
    K[审计日志] --> H
```

1. **最小权限原则 (Principle of Least Privilege)**：每个组件、用户和服务只拥有完成其功能所需的最小权限
2. **零信任架构 (Zero Trust Architecture)**：默认不信任任何实体，所有访问都需要验证
3. **深度防御 (Defense in Depth)**：在多个层次实施安全控制
4. **安全左移 (Shift Left Security)**：在开发早期阶段就考虑安全问题

### Kubernetes安全威胁模型

```mermaid
graph LR
    A[外部攻击者] --> B[API Server]
    C[恶意内部用户] --> D[Etcd]
    E[供应链攻击] --> F[容器镜像]
    G[网络攻击者] --> H[Pod间通信]
    I[特权用户滥用] --> J[集群节点]
    
    B --> K[认证绕过]
    D --> K
    F --> L[恶意代码执行]
    H --> M[服务拒绝]
    J --> N[容器逃逸]
```

#### 主要威胁类型：
- **认证绕过 (Authentication Bypass)**：试图绕过身份认证机制
- **特权提升 (Privilege Escalation)**：获得超出应有权限的访问
- **横向移动 (Lateral Movement)**：在集群内部扩大攻击范围
- **数据泄露 (Data Breach)**：未经授权访问敏感数据
- **服务拒绝 (Denial of Service)**：破坏服务可用性
- **容器逃逸 (Container Escape)**：突破容器隔离获得宿主机权限

## 安全威胁分析

### 威胁分类与影响评估

```mermaid
graph TB
    A[Kubernetes威胁模型] --> B[外部威胁]
    A --> C[内部威胁]
    A --> D[供应链威胁]
    A --> E[配置错误]
    
    B --> B1[网络扫描]
    B --> B2[API攻击]
    B --> B3[DDoS攻击]
    
    C --> C1[恶意用户]
    C --> C2[被攻陷账户]
    C --> C3[特权滥用]
    
    D --> D1[恶意镜像]
    D --> D2[污染的依赖]
    D --> D3[恶意插件]
    
    E --> E1[默认配置]
    E --> E2[权限过度宽松]
    E --> E3[网络策略缺失]
```

### 攻击向量分析

#### 1. API Server攻击
```python
class APIServerAttackVector:
    """API Server攻击向量分析"""
    
    def __init__(self):
        self.common_attacks = {
            'authentication_bypass': {
                'description': '绕过身份认证',
                'impact': 'critical',
                'examples': [
                    '使用无效令牌获得访问权限',
                    '利用配置错误跳过认证步骤'
                ],
                'mitigation': [
                    '强制启用认证机制',
                    '定期轮换认证令牌',
                    '监控异常认证模式'
                ]
            },
            'authorization_escalation': {
                'description': '权限提升攻击',
                'impact': 'high',
                'examples': [
                    '滥用ServiceAccount权限',
                    '利用RBAC配置错误'
                ],
                'mitigation': [
                    '实施最小权限原则',
                    '定期审计RBAC配置',
                    '使用Pod Security Standards'
                ]
            },
            'api_abuse': {
                'description': 'API滥用',
                'impact': 'medium',
                'examples': [
                    '过度频繁的API调用',
                    '异常大的资源请求'
                ],
                'mitigation': [
                    '配置API限流',
                    '实施请求配额',
                    '监控API使用模式'
                ]
            }
        }
    
    def assess_attack_surface(self):
        """评估攻击面"""
        attack_surface = {
            'exposed_endpoints': [
                '/api/v1/* - 核心Kubernetes API',
                '/apis/* - 扩展API',
                '/metrics - 监控指标',
                '/healthz - 健康检查'
            ],
            'authentication_mechanisms': [
                'Client Certificates',
                'Bearer Tokens',
                'Basic Authentication',
                'Webhook Token Authentication',
                'OpenID Connect'
            ],
            'authorization_modes': [
                'RBAC (Role-Based Access Control)',
                'ABAC (Attribute-Based Access Control)',
                'Node Authorization',
                'Webhook Authorization'
            ]
        }
        return attack_surface
```

#### 2. 容器逃逸攻击
```mermaid
graph LR
    A[容器环境] --> B[内核漏洞]
    A --> C[特权容器]
    A --> D[危险挂载]
    A --> E[共享命名空间]
    
    B --> F[容器逃逸]
    C --> F
    D --> F
    E --> F
    
    F --> G[宿主系统访问]
    G --> H[其他容器访问]
    H --> I[集群横向移动]
```

## 多层安全架构

### Kubernetes安全架构图

```mermaid
graph TB
    subgraph "第1层：基础设施安全"
        A1[集群节点加固]
        A2[网络安全分段]
        A3[访问控制]
    end
    
    subgraph "第2层：集群安全"
        B1[API Server安全]
        B2[Etcd加密]
        B3[集群组件加固]
    end
    
    subgraph "第3层：命名空间隔离"
        C1[网络策略]
        C2[资源配额]
        C3[权限隔离]
    end
    
    subgraph "第4层：应用安全"
        D1[Pod安全上下文]
        D2[容器安全配置]
        D3[ServiceAccount]
    end
    
    subgraph "第5层：数据安全"
        E1[静态加密]
        E2[传输加密]
        E3[密钥管理]
    end
    
    subgraph "第6层：监控审计"
        F1[安全事件监控]
        F2[审计日志]
        F3[合规检查]
    end
    
    A1 --> B1
    A2 --> B2
    B1 --> C1
    B2 --> C2
    C1 --> D1
    C2 --> D2
    D1 --> E1
    D2 --> E2
    E1 --> F1
    E2 --> F2
```

### 安全组件实现

```python
class KubernetesSecurityArchitecture:
    """Kubernetes安全架构实现"""
    
    def __init__(self):
        self.security_layers = self._initialize_security_layers()
        self.security_policies = self._initialize_policies()
        self.monitoring_components = self._initialize_monitoring()
    
    def _initialize_security_layers(self):
        """初始化安全层级"""
        return {
            'infrastructure': {
                'name': '基础设施安全层',
                'components': [
                    '节点操作系统加固',
                    '网络安全配置',
                    '物理安全控制',
                    '访问控制机制'
                ],
                'controls': {
                    'os_hardening': {
                        'principle': '操作系统最小化配置',
                        'implementations': [
                            '禁用不必要的服务',
                            '配置防火墙规则',
                            '启用审计日志',
                            '定期安全更新'
                        ]
                    },
                    'network_security': {
                        'principle': '网络分段和隔离',
                        'implementations': [
                            'VPC网络配置',
                            '安全组规则',
                            '网络访问控制列表',
                            '加密通信通道'
                        ]
                    }
                }
            },
            'cluster': {
                'name': '集群安全层',
                'components': [
                    'API Server安全配置',
                    'Etcd数据加密',
                    '集群组件认证',
                    '集群网络安全'
                ],
                'controls': {
                    'api_security': {
                        'principle': 'API访问控制和验证',
                        'configurations': [
                            '--authentication-mode=RBAC',
                            '--authorization-mode=Node,RBAC',
                            '--enable-admission-plugins=PodSecurityPolicy',
                            '--audit-log-path=/var/log/kubernetes/audit.log'
                        ]
                    },
                    'etcd_encryption': {
                        'principle': '静态数据加密',
                        'requirements': [
                            '配置EncryptionProvider',
                            '使用强加密算法',
                            '定期轮换加密密钥',
                            '备份加密配置'
                        ]
                    }
                }
            },
            'namespace': {
                'name': '命名空间隔离层',
                'components': [
                    '命名空间资源隔离',
                    '网络策略控制',
                    'RBAC权限隔离',
                    '资源配额管理'
                ],
                'controls': {
                    'resource_isolation': {
                        'principle': '资源使用限制',
                        'mechanisms': [
                            'ResourceQuota - 资源总量限制',
                            'LimitRange - 单Pod资源限制',
                            'NetworkPolicy - 网络访问控制',
                            'PodSecurityContext - Pod级别隔离'
                        ]
                    }
                }
            },
            'application': {
                'name': '应用安全层',
                'components': [
                    'Pod安全上下文',
                    '容器安全配置',
                    'ServiceAccount管理',
                    '密钥和配置管理'
                ],
                'controls': {
                    'pod_security': {
                        'principle': '容器运行安全配置',
                        'requirements': [
                            'runAsNonRoot=true',
                            'allowPrivilegeEscalation=false',
                            'readOnlyRootFilesystem=true',
                            'capabilities.drop=ALL'
                        ]
                    },
                    'service_account': {
                        'principle': '最小权限ServiceAccount',
                        'best_practices': [
                            '为每个应用创建专用ServiceAccount',
                            '避免使用默认ServiceAccount',
                            '定期审查ServiceAccount权限',
                            '使用Token投影功能'
                        ]
                    }
                }
            }
        }
    
    def _initialize_policies(self):
        """初始化安全策略"""
        return {
            'pod_security_policy': {
                'name': 'Pod安全策略',
                'levels': {
                    'privileged': {
                        'description': '特权级别 - 完全访问权限',
                        'restrictions': '无限制',
                        'use_cases': '仅用于系统组件'
                    },
                    'baseline': {
                        'description': '基线级别 - 基本安全控制',
                        'restrictions': [
                            '禁止特权容器',
                            '禁用-hostPath挂载',
                            '要求非root用户运行'
                        ],
                        'use_cases': '大多数应用程序'
                    },
                    'restricted': {
                        'description': '严格级别 - 最高安全限制',
                        'restrictions': [
                            '所有基线限制',
                            '只读根文件系统',
                            '禁用所有capabilities',
                            '强制SecurityContext'
                        ],
                        'use_cases': '高安全要求应用'
                    }
                }
            },
            'network_policy': {
                'name': '网络策略',
                'types': {
                    'deny_all': {
                        'description': '拒绝所有流量',
                        'configuration': {
                            'podSelector': {},
                            'policyTypes': ['Ingress', 'Egress']
                        }
                    },
                    'allow_specific': {
                        'description': '允许特定流量',
                        'configuration': {
                            'podSelector': {},
                            'ingress': [{'from': [{'podSelector': {'app': 'allowed-app'}}]}],
                            'egress': [{'to': [{'podSelector': {'app': 'external-service'}}]}]
                        }
                    }
                }
            }
        }
    
    def demonstrate_security_architecture(self):
        """演示安全架构"""
        print("=== Kubernetes多层安全架构演示 ===")
        
        # 显示各层安全组件
        for layer_name, layer_info in self.security_layers.items():
            print(f"\n{layer_info['name']}:")
            for component in layer_info['components']:
                print(f"  - {component}")
        
        # 显示安全策略
        print("\n安全策略配置:")
        for policy_name, policy_info in self.security_policies.items():
            print(f"  {policy_info['name']}:")
            if policy_name == 'pod_security_policy':
                for level, details in policy_info['levels'].items():
                    print(f"    {level}: {details['description']}")
        
        # 安全检查清单
        print("\n安全检查清单:")
        checklist = [
            "✓ 启用RBAC并配置最小权限",
            "✓ 配置Pod Security Standards",
            "✓ 实施网络策略进行微服务隔离",
            "✓ 启用Etcd加密",
            "✓ 配置镜像拉取策略",
            "✓ 实施安全上下文",
            "✓ 启用审计日志",
            "✓ 配置资源配额",
            "✓ 定期安全扫描",
            "✓ 监控安全事件"
        ]
        
        for item in checklist:
            print(f"  {item}")
```

## 身份认证与授权

### 认证与授权原理

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API Server
    participant R as RBAC
    participant S as ServiceAccount
    participant E as Etcd
    
    C->>A: 发送API请求(包含认证信息)
    A->>A: 验证客户端认证
    A->>R: 检查RBAC权限
    R->>R: 验证用户/ServiceAccount权限
    R->>A: 返回权限决策
    A->>E: 执行授权的操作
    E->>A: 返回结果
    A->>C: 返回响应
```

### RBAC实现详解

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ResourceType(Enum):
    """Kubernetes资源类型"""
    POD = "pods"
    SERVICE = "services"
    CONFIGMAP = "configmaps"
    SECRET = "secrets"
    DEPLOYMENT = "deployments"
    STATEFULSET = "statefulsets"
    DAEMONSET = "daemonsets"
    INGRESS = "ingresses"
    NETWORKPOLICY = "networkpolicies"
    ROLE = "roles"
    ROLEBINDING = "rolebindings"
    CLUSTERROLE = "clusterroles"
    CLUSTERROLEBINDING = "clusterrolebindings"

class VerbType(Enum):
    """Kubernetes操作动词"""
    GET = "get"
    LIST = "list"
    WATCH = "watch"
    CREATE = "create"
    UPDATE = "update"
    PATCH = "patch"
    DELETE = "delete"
    DELETECOLLECTION = "deletecollection"

@dataclass
class RBACRule:
    """RBAC规则定义"""
    api_groups: List[str]
    resources: List[str]
    resource_names: Optional[List[str]] = None
    verbs: List[str] = None

@dataclass
class Role:
    """角色定义"""
    name: str
    namespace: str
    rules: List[RBACRule]
    description: str = ""

@dataclass
class ServiceAccountInfo:
    """ServiceAccount信息"""
    name: str
    namespace: str
    tokens: List[str]
    bound_pods: List[str]
    assigned_roles: List[str]

class RBACManager:
    """RBAC权限管理器"""
    
    def __init__(self):
        self.roles = {}
        self.cluster_roles = {}
        self.role_bindings = {}
        self.cluster_role_bindings = {}
        self.service_accounts = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """初始化默认角色"""
        # 只读角色
        read_only_role = Role(
            name="read-only",
            namespace="default",
            rules=[
                RBACRule(
                    api_groups=[""],
                    resources=["pods", "services", "configmaps"],
                    verbs=["get", "list", "watch"]
                ),
                RBACRule(
                    api_groups=["apps"],
                    resources=["deployments", "replicasets"],
                    verbs=["get", "list", "watch"]
                )
            ],
            description="只读权限，可以查看资源"
        )
        
        # 开发者角色
        developer_role = Role(
            name="developer",
            namespace="default",
            rules=[
                RBACRule(
                    api_groups=[""],
                    resources=["pods", "services", "configmaps", "secrets"],
                    verbs=["get", "list", "create", "update", "patch"]
                ),
                RBACRule(
                    api_groups=["apps"],
                    resources=["deployments", "replicasets"],
                    verbs=["get", "list", "create", "update", "patch"]
                )
            ],
            description="开发人员权限，可以部署和管理应用"
        )
        
        # 运维人员角色
        ops_role = Role(
            name="ops",
            namespace="default",
            rules=[
                RBACRule(
                    api_groups=[""],
                    resources=["*"],
                    verbs=["*"]
                ),
                RBACRule(
                    api_groups=["apps"],
                    resources=["*"],
                    verbs=["*"]
                )
            ],
            description="运维人员完整权限"
        )
        
        self.roles = {
            "read-only": read_only_role,
            "developer": developer_role,
            "ops": ops_role
        }
    
    def create_namespace_admin_role(self, namespace: str, 
                                   limited_resources: List[str]) -> Role:
        """创建命名空间管理员角色"""
        admin_role = Role(
            name=f"admin-{namespace}",
            namespace=namespace,
            rules=[
                RBACRule(
                    api_groups=[""],
                    resources=limited_resources,
                    verbs=["*"]
                )
            ],
            description=f"{namespace}命名空间管理员权限"
        )
        
        self.roles[f"admin-{namespace}"] = admin_role
        return admin_role
    
    def create_service_account_with_minimal_permissions(self, 
                                                       app_name: str, 
                                                       namespace: str,
                                                       required_permissions: List[str]) -> ServiceAccountInfo:
        """创建最小权限ServiceAccount"""
        # 创建ServiceAccount
        sa = ServiceAccountInfo(
            name=f"{app_name}-sa",
            namespace=namespace,
            tokens=[],  # 实际环境中会有token
            bound_pods=[app_name],
            assigned_roles=required_permissions
        )
        
        # 创建专用的Role
        app_role = Role(
            name=f"{app_name}-role",
            namespace=namespace,
            rules=[
                RBACRule(
                    api_groups=[""],
                    resources=["configmaps", "secrets"],
                    verbs=["get", "list"]
                ),
                RBACRule(
                    api_groups=["apps"],
                    resources=["deployments"],
                    verbs=["get", "list", "watch"]
                )
            ],
            description=f"{app_name}应用专用角色"
        )
        
        self.roles[f"{app_name}-role"] = app_role
        self.service_accounts[f"{app_name}-sa"] = sa
        
        return sa
    
    def audit_permissions(self, user_type: str) -> Dict[str, Any]:
        """权限审计"""
        audit_result = {
            'user_type': user_type,
            'excessive_permissions': [],
            'missing_permissions': [],
            'recommendations': []
        }
        
        if user_type == "service_account":
            # 检查ServiceAccount权限
            for sa_name, sa_info in self.service_accounts.items():
                if len(sa_info.assigned_roles) > 3:
                    audit_result['excessive_permissions'].append({
                        'service_account': sa_name,
                        'reason': 'Assigned too many roles',
                        'assigned_roles': sa_info.assigned_roles
                    })
        
        # 生成建议
        audit_result['recommendations'] = [
            "定期审查用户和ServiceAccount权限",
            "移除不再使用的角色绑定",
            "使用命名空间隔离权限",
            "实施权限审批流程"
        ]
        
        return audit_result
    
    def generate_yamls(self, role_name: str) -> Dict[str, str]:
        """生成Kubernetes YAML配置"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} not found")
        
        role = self.roles[role_name]
        
        # 生成Role YAML
        role_yaml = f"""apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {role.name}
  namespace: {role.namespace}
rules:
"""
        for rule in role.rules:
            role_yaml += f"""- apiGroups: {json.dumps(rule.api_groups)}
  resources: {json.dumps(rule.resources)}
  verbs: {json.dumps(rule.verbs)}
"""
        
        # 生成RoleBinding YAML
        role_binding_yaml = f"""apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {role.name}-binding
  namespace: {role.namespace}
subjects:
- kind: User
  name: user-{role.name}
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: {role.name}
  apiGroup: rbac.authorization.k8s.io
"""
        
        return {
            'role': role_yaml,
            'role_binding': role_binding_yaml
        }

# 演示RBAC管理
def demo_rbac_management():
    """演示RBAC管理功能"""
    print("=== Kubernetes RBAC管理演示 ===")
    
    # 创建RBAC管理器
    rbac_mgr = RBACManager()
    
    # 创建应用ServiceAccount
    sa = rbac_mgr.create_service_account_with_minimal_permissions(
        app_name="web-app",
        namespace="production",
        required_permissions=["web-app-role"]
    )
    
    print(f"创建ServiceAccount: {sa.name}")
    print(f"命名空间: {sa.namespace}")
    print(f"绑定应用: {sa.bound_pods}")
    
    # 生成YAML配置
    yamls = rbac_mgr.generate_yamls("web-app-role")
    print("\n生成的Role YAML:")
    print(yamls['role'])
    print("\n生成的RoleBinding YAML:")
    print(yamls['role_binding'])
    
    # 权限审计
    audit = rbac_mgr.audit_permissions("service_account")
    print("\n权限审计结果:")
    print(f"审计用户类型: {audit['user_type']}")
    if audit['excessive_permissions']:
        for issue in audit['excessive_permissions']:
            print(f"  权限过度: {issue}")
    
    # 角色权限对比
    print("\n角色权限对比:")
    roles = ['read-only', 'developer', 'ops']
    for role_name in roles:
        if role_name in rbac_mgr.roles:
            role = rbac_mgr.roles[role_name]
            print(f"  {role_name}: {role.description}")
            for rule in role.rules:
                print(f"    资源: {rule.resources}, 操作: {rule.verbs}")

# 运行演示
demo_rbac_management()
```

## 网络安全策略

### 网络安全架构原理

```mermaid
graph TB
    subgraph "外部网络"
        A[Internet]
    end
    
    subgraph "Kubernetes集群网络"
        B[Ingress Controller]
        C[Service Mesh]
        D[Pod间通信]
        E[外部服务访问]
    end
    
    subgraph "网络策略层级"
        F[集群级策略]
        G[命名空间级策略]
        H[Pod级策略]
    end
    
    A --> B
    B --> C
    C --> D
    C --> E
    
    F --> G
    G --> H
    
    D --> I[网络隔离]
    E --> J[访问控制]
    
    I --> K[零信任网络]
    J --> K
```

### 网络策略实现

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class NetworkPolicyType(Enum):
    """网络策略类型"""
    INGRESS = "Ingress"
    EGRESS = "Egress"
    BOTH = "Both"

class Protocol(Enum):
    """网络协议"""
    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"

@dataclass
class NetworkRule:
    """网络规则定义"""
    from_sources: Optional[List[Dict[str, Any]]] = None
    to_sources: Optional[List[Dict[str, Any]]] = None
    ports: Optional[List[Dict[str, Any]]] = None
    protocols: Optional[List[Dict[str, Any]]] = None

class NetworkPolicyManager:
    """网络策略管理器"""
    
    def __init__(self):
        self.policies = {}
        self.namespace_policies = {}
        self.service_mesh_config = {}
    
    def create_microservice_isolation(self, namespace: str, services: Dict[str, Dict]) -> Dict[str, Any]:
        """创建微服务网络隔离策略"""
        policy_config = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': f'microservice-isolation-{namespace}',
                'namespace': namespace,
                'labels': {
                    'app': 'microservice-isolation',
                    'purpose': 'security'
                }
            },
            'spec': {
                'podSelector': {},  # 选择命名空间中的所有Pod
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [],
                'egress': []
            }
        }
        
        # 为每个服务配置入站规则
        for service_name, service_config in services.items():
            # 入站规则：只允许来自指定来源的流量
            ingress_rule = {
                'from': [{
                    'podSelector': {
                        'matchLabels': {
                            'app': service_name
                        }
                    }
                }],
                'ports': [{
                    'protocol': 'TCP',
                    'port': service_config.get('port', 8080)
                }]
            }
            policy_config['spec']['ingress'].append(ingress_rule)
            
            # 出站规则：允许访问指定外部服务
            egress_rule = {
                'to': [],  # 外部服务
                'ports': [{
                    'protocol': 'TCP',
                    'port': 443  # HTTPS
                }]
            }
            policy_config['spec']['egress'].append(egress_rule)
        
        return policy_config
    
    def create_three_tier_architecture_policy(self, namespace: str) -> Dict[str, Any]:
        """创建三层架构网络策略"""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'three-tier-architecture',
                'namespace': namespace
            },
            'spec': {
                'podSelector': {},
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [
                    # 允许外部访问前端层
                    {
                        'from': [],
                        'ports': [{'protocol': 'TCP', 'port': 80}]
                    },
                    # 前端层可以访问后端层
                    {
                        'from': [{'podSelector': {'tier': 'frontend'}}],
                        'ports': [{'protocol': 'TCP', 'port': 8080}]
                    },
                    # 后端层可以访问数据层
                    {
                        'from': [{'podSelector': {'tier': 'backend'}}],
                        'ports': [{'protocol': 'TCP', 'port': 3306}]
                    }
                ],
                'egress': [
                    # 允许后端层访问数据库
                    {
                        'to': [{'podSelector': {'tier': 'database'}}],
                        'ports': [{'protocol': 'TCP', 'port': 3306}]
                    }
                ]
            }
        }
    
    def create_ZeroTrust_policy(self, namespace: str, app_name: str) -> Dict[str, Any]:
        """创建零信任网络策略"""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': f'zerotrust-{app_name}',
                'namespace': namespace
            },
            'spec': {
                'podSelector': {
                    'matchLabels': {'app': app_name}
                },
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [
                    # 明确允许的入站流量
                    {
                        'from': [{
                            'namespaceSelector': {
                                'matchLabels': {'trust-level': 'verified'}
                            }
                        }],
                        'ports': [{'protocol': 'TCP', 'port': 8080}]
                    }
                ],
                'egress': [
                    # 明确允许的出站流量
                    {
                        'to': [{
                            'namespaceSelector': {
                                'matchLabels': {'trust-level': 'verified'}
                            }
                        }],
                        'ports': [{'protocol': 'TCP', 'port': 443}]
                    },
                    # DNS查询
                    {
                        'to': [],
                        'ports': [{'protocol': 'UDP', 'port': 53}]
                    }
                ]
            }
        }
    
    def generate_comprehensive_network_policies(self, namespace: str) -> List[Dict[str, Any]]:
        """生成综合网络策略集"""
        policies = []
        
        # 1. 默认拒绝所有流量
        deny_all_policy = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'default-deny-all',
                'namespace': namespace
            },
            'spec': {
                'podSelector': {},
                'policyTypes': ['Ingress', 'Egress']
            }
        }
        policies.append(deny_all_policy)
        
        # 2. 允许DNS查询
        dns_policy = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'allow-dns',
                'namespace': namespace
            },
            'spec': {
                'podSelector': {},
                'policyTypes': ['Egress'],
                'egress': [{
                    'to': [],
                    'ports': [{'protocol': 'UDP', 'port': 53}]
                }]
            }
        }
        policies.append(dns_policy)
        
        # 3. 允许健康检查
        health_check_policy = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'allow-health-checks',
                'namespace': namespace
            },
            'spec': {
                'podSelector': {},
                'policyTypes': ['Ingress'],
                'ingress': [{
                    'from': [{
                        'namespaceSelector': {
                            'matchLabels': {'name': 'kubernetes-dashboard'}
                        }
                    }],
                    'ports': [{'protocol': 'TCP', 'port': 8080}]
                }]
            }
        }
        policies.append(health_check_policy)
        
        return policies

# Service Mesh安全配置
class ServiceMeshSecurity:
    """服务网格安全配置"""
    
    def __init__(self):
        self.mtls_config = {}
        self.traffic_policies = {}
        self.security_policies = {}
    
    def configure_mtls(self, namespace: str, services: List[str]) -> Dict[str, Any]:
        """配置双向TLS认证"""
        return {
            'apiVersion': 'security.istio.io/v1beta1',
            'kind': 'PeerAuthentication',
            'metadata': {
                'name': 'default',
                'namespace': namespace
            },
            'spec': {
                'mtls': {
                    'mode': 'STRICT'  # 强制mTLS
                }
            }
        }
    
    def create_traffic_policy(self, namespace: str, service_name: str) -> Dict[str, Any]:
        """创建流量安全策略"""
        return {
            'apiVersion': 'networking.istio.io/v1alpha3',
            'kind': 'DestinationRule',
            'metadata': {
                'name': f'{service_name}-security-policy',
                'namespace': namespace
            },
            'spec': {
                'host': f'{service_name}.{namespace}.svc.cluster.local',
                'trafficPolicy': {
                    'tls': {
                        'mode': 'ISTIO_MUTUAL'  # 自动mTLS
                    },
                    'connectionPool': {
                        'tcp': {
                            'maxConnections': 100
                        },
                        'http': {
                            'http1MaxPendingRequests': 10,
                            'maxRequestsPerConnection': 2
                        }
                    },
                    'outlierDetection': {
                        'consecutiveErrors': 3,
                        'interval': 30,
                        'baseEjectionTime': 30
                    }
                }
            }
        }
    
    def demonstrate_network_security(self):
        """演示网络安全配置"""
        print("=== Kubernetes网络安全策略演示 ===")
        
        network_mgr = NetworkPolicyManager()
        service_mesh = ServiceMeshSecurity()
        
        # 微服务架构
        services = {
            'user-service': {'port': 8080},
            'order-service': {'port': 8081},
            'payment-service': {'port': 8082}
        }
        
        # 创建微服务隔离策略
        isolation_policy = network_mgr.create_microservice_isolation('production', services)
        print(f"微服务隔离策略: {isolation_policy['metadata']['name']}")
        print(f"入站规则数量: {len(isolation_policy['spec']['ingress'])}")
        print(f"出站规则数量: {len(isolation_policy['spec']['egress'])}")
        
        # 创建三层架构策略
        three_tier_policy = network_mgr.create_three_tier_architecture_policy('production')
        print(f"\n三层架构策略: {three_tier_policy['metadata']['name']}")
        print(f"入站规则数量: {len(three_tier_policy['spec']['ingress'])}")
        
        # 生成综合策略集
        comprehensive_policies = network_mgr.generate_comprehensive_network_policies('production')
        print(f"\n综合网络策略集 ({len(comprehensive_policies)} 个策略):")
        for policy in comprehensive_policies:
            print(f"  - {policy['metadata']['name']}")
        
        # Service Mesh配置
        mtls_config = service_mesh.configure_mtls('production', ['user-service', 'order-service'])
        print(f"\nmTLS配置: {mtls_config['metadata']['name']}")
        print(f"认证模式: {mtls_config['spec']['mtls']['mode']}")
        
        traffic_policy = service_mesh.create_traffic_policy('production', 'user-service')
        print(f"\n流量安全策略: {traffic_policy['metadata']['name']}")
        print(f"TLS模式: {traffic_policy['spec']['trafficPolicy']['tls']['mode']}")

# 运行演示
demonstrate_network_security()
```

## Pod与容器安全

### 容器安全架构原理

```mermaid
graph TB
    subgraph "Pod安全上下文"
        A[运行用户配置] --> F[容器安全控制]
        B[文件系统权限] --> F
        C[能力(Capabilities)] --> F
        D[SELinux/AppArmor] --> F
        E[安全策略] --> F
    end
    
    subgraph "容器运行时安全"
        G[命名空间隔离] --> H[资源限制]
        I[控制组(cgroups)] --> H
        J[只读挂载] --> K[文件完整性]
        L[临时文件系统] --> K
    end
    
    subgraph "镜像安全"
        M[基础镜像安全] --> N[供应链安全]
        O[漏洞扫描] --> N
        P[镜像签名] --> N
    end
    
    F --> G
    K --> M
    
    H --> O
    N --> P
```

### Pod Security Standards实现

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class PodSecurityLevel(Enum):
    """Pod安全级别"""
    PRIVILEGED = "privileged"
    BASELINE = "baseline"
    RESTRICTED = "restricted"

class SecurityContext(Enum):
    """安全上下文配置"""
    RUN_AS_NON_ROOT = "runAsNonRoot"
    RUN_AS_USER = "runAsUser"
    READ_ONLY_ROOT_FILESYSTEM = "readOnlyRootFilesystem"
    ALLOW_PRIVILEGE_ESCALATION = "allowPrivilegeEscalation"
    DROP_CAPABILITIES = "capabilities.drop"
    ADD_CAPABILITIES = "capabilities.add"

@dataclass
class SecurityRequirement:
    """安全要求定义"""
    requirement: str
    value: Any
    description: str
    violation_action: str = "reject"  # reject, warn, audit

class PodSecurityManager:
    """Pod安全管理器"""
    
    def __init__(self):
        self.security_levels = self._initialize_security_levels()
        self.container_hardening = {}
        self.runtime_configs = {}
    
    def _initialize_security_levels(self):
        """初始化Pod安全级别配置"""
        return {
            PodSecurityLevel.PRIVILEGED: {
                'name': '特权级别',
                'description': '完全访问权限，仅用于系统组件',
                'restrictions': {
                    'allowPrivilegeEscalation': None,  # 不限制
                    'capabilities': {
                        'add': ['*'],  # 允许所有capabilities
                        'drop': []
                    },
                    'readOnlyRootFilesystem': False,  # 允许写根文件系统
                    'runAsNonRoot': False,  # 允许root用户
                    'volumes': ['*']  # 允许所有挂载类型
                },
                'use_cases': ['网络插件', '存储插件', '监控系统']
            },
            PodSecurityLevel.BASELINE: {
                'name': '基线级别',
                'description': '基本安全控制，适用于大多数应用',
                'restrictions': {
                    'allowPrivilegeEscalation': False,
                    'capabilities': {
                        'add': [],
                        'drop': ['ALL']
                    },
                    'readOnlyRootFilesystem': False,
                    'runAsNonRoot': True,
                    'volumes': ['configMap', 'emptyDir', 'projected', 'secret', 'downwardAPI', 'persistentVolumeClaim']
                },
                'use_cases': ['Web应用', 'API服务', '微服务']
            },
            PodSecurityLevel.RESTRICTED: {
                'name': '严格级别',
                'description': '最高安全限制，适用于高安全要求应用',
                'restrictions': {
                    'allowPrivilegeEscalation': False,
                    'capabilities': {
                        'add': [],
                        'drop': ['ALL']
                    },
                    'readOnlyRootFilesystem': True,
                    'runAsNonRoot': True,
                    'runAsUser': None,  # 强制指定非root用户
                    'fsGroup': None,    # 强制指定文件系统组
                    'seccompProfile': 'RuntimeDefault',
                    'volumes': ['configMap', 'secret', 'projected', 'downwardAPI']
                },
                'use_cases': ['金融应用', '政府系统', '医疗应用']
            }
        }
    
    def generate_secure_pod_spec(self, 
                                app_name: str,
                                image: str,
                                security_level: PodSecurityLevel,
                                namespace: str = 'default',
                                run_as_user: int = 1000,
                                **custom_config) -> Dict[str, Any]:
        """生成安全的Pod规格"""
        
        if security_level not in self.security_levels:
            raise ValueError(f"Unsupported security level: {security_level}")
        
        level_config = self.security_levels[security_level]
        restrictions = level_config['restrictions']
        
        # 构建安全上下文
        pod_security_context = {
            'runAsNonRoot': restrictions['runAsNonRoot'],
            'runAsUser': run_as_user if restrictions['runAsUser'] is None else restrictions['runAsUser'],
            'fsGroup': 1000 if 'fsGroup' not in restrictions else restrictions['fsGroup']
        }
        
        # 如果是严格级别，添加seccomp配置
        if security_level == PodSecurityLevel.RESTRICTED:
            pod_security_context['seccompProfile'] = {
                'type': 'RuntimeDefault'
            }
        
        # 构建容器安全上下文
        container_security_context = {
            'runAsNonRoot': restrictions['runAsNonRoot'],
            'allowPrivilegeEscalation': restrictions['allowPrivilegeEscalation'],
            'readOnlyRootFilesystem': restrictions['readOnlyRootFilesystem'],
            'capabilities': {
                'drop': restrictions['capabilities']['drop'],
                'add': restrictions['capabilities']['add'] if 'add' in restrictions['capabilities'] else []
            }
        }
        
        # 如果是严格级别，添加seccomp配置
        if security_level == PodSecurityLevel.RESTRICTED:
            container_security_context['seccompProfile'] = {
                'type': 'RuntimeDefault'
            }
        
        # 构建Pod规格
        pod_spec = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': app_name,
                'namespace': namespace,
                'labels': {
                    'app': app_name,
                    'security-level': security_level.value,
                    'pod-security.kubernetes.io/enforce': security_level.value
                },
                'annotations': {
                    'container.apparmor.security.beta.kubernetes.io/' + app_name: 'runtime/default',
                    'security.kubernetes.io/podSecurityPolicy': security_level.value
                }
            },
            'spec': {
                'securityContext': pod_security_context,
                'containers': [{
                    'name': app_name,
                    'image': image,
                    'securityContext': container_security_context,
                    'volumeMounts': self._generate_volume_mounts(restrictions),
                    'resources': {
                        'requests': {
                            'cpu': '100m',
                            'memory': '128Mi'
                        },
                        'limits': {
                            'cpu': '500m',
                            'memory': '512Mi'
                        }
                    },
                    'livenessProbe': {
                        'exec': {
                            'command': ['/bin/sh', '-c', 'pgrep ' + app_name]
                        },
                        'initialDelaySeconds': 30,
                        'periodSeconds': 10
                    },
                    'readinessProbe': {
                        'exec': {
                            'command': ['/bin/sh', '-c', 'wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1']
                        },
                        'initialDelaySeconds': 5,
                        'periodSeconds': 5
                    }
                }],
                'volumes': self._generate_volumes(restrictions),
                'dnsPolicy': 'ClusterFirst',
                'restartPolicy': 'Always'
            }
        }
        
        # 添加自定义配置
        pod_spec['spec'].update(custom_config)
        
        return pod_spec
    
    def _generate_volume_mounts(self, restrictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成安全卷挂载"""
        allowed_volumes = restrictions.get('volumes', [])
        volume_mounts = []
        
        if 'configMap' in allowed_volumes:
            volume_mounts.append({
                'name': 'config-volume',
                'mountPath': '/etc/config',
                'readOnly': True
            })
        
        if 'secret' in allowed_volumes:
            volume_mounts.append({
                'name': 'secret-volume',
                'mountPath': '/etc/secrets',
                'readOnly': True
            })
        
        if 'emptyDir' in allowed_volumes:
            volume_mounts.append({
                'name': 'tmp-volume',
                'mountPath': '/tmp',
                'readOnly': False
            })
        
        return volume_mounts
    
    def _generate_volumes(self, restrictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成安全卷配置"""
        allowed_volumes = restrictions.get('volumes', [])
        volumes = []
        
        if 'configMap' in allowed_volumes:
            volumes.append({
                'name': 'config-volume',
                'configMap': {
                    'name': 'app-config'
                }
            })
        
        if 'secret' in allowed_volumes:
            volumes.append({
                'name': 'secret-volume',
                'secret': {
                    'secretName': 'app-secrets'
                }
            })
        
        if 'emptyDir' in allowed_volumes:
            volumes.append({
                'name': 'tmp-volume',
                'emptyDir': {}
            })
        
        if 'downwardAPI' in allowed_volumes:
            volumes.append({
                'name': 'downward-api-volume',
                'downwardAPI': {
                    'items': [{
                        'path': 'namespace',
                        'fieldRef': {
                            'fieldPath': 'metadata.namespace'
                        }
                    }]
                }
            })
        
        return volumes
    
    def validate_security_compliance(self, pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """验证Pod安全合规性"""
        compliance_result = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        security_context = pod_spec.get('spec', {}).get('securityContext', {})
        containers = pod_spec.get('spec', {}).get('containers', [])
        
        # 检查Pod级别安全上下文
        if not security_context.get('runAsNonRoot'):
            compliance_result['violations'].append({
                'level': 'high',
                'component': 'pod',
                'issue': 'Container should not run as root',
                'remediation': 'Set runAsNonRoot: true in pod securityContext'
            })
        
        # 检查容器级别安全上下文
        for container in containers:
            container_security = container.get('securityContext', {})
            
            # 检查权限提升
            if container_security.get('allowPrivilegeEscalation', True):
                compliance_result['violations'].append({
                    'level': 'medium',
                    'component': 'container',
                    'container': container['name'],
                    'issue': 'Privilege escalation is allowed',
                    'remediation': 'Set allowPrivilegeEscalation: false'
                })
            
            # 检查根文件系统
            if not container_security.get('readOnlyRootFilesystem'):
                compliance_result['warnings'].append({
                    'level': 'low',
                    'container': container['name'],
                    'issue': 'Root filesystem is writable',
                    'recommendation': 'Consider using readOnlyRootFilesystem: true'
                })
            
            # 检查capabilities
            capabilities = container_security.get('capabilities', {})
            dropped_caps = capabilities.get('drop', [])
            if 'ALL' not in dropped_caps:
                compliance_result['warnings'].append({
                    'level': 'medium',
                    'container': container['name'],
                    'issue': 'Not all capabilities are dropped',
                    'recommendation': 'Drop all unnecessary capabilities'
                })
        
        # 生成建议
        if compliance_result['violations']:
            compliance_result['compliant'] = False
        
        compliance_result['recommendations'] = [
            '定期更新容器镜像以修复安全漏洞',
            '使用镜像签名验证容器完整性',
            '监控容器运行时异常行为',
            '实施资源限制防止拒绝服务攻击',
            '配置适当的健康检查和探针'
        ]
        
        return compliance_result

class RuntimeSecurity:
    """容器运行时安全"""
    
    def __init__(self):
        self.runtime_profiles = {}
        self.sandbox_configs = {}
    
    def generate_runtime_class_config(self, runtime_name: str) -> Dict[str, Any]:
        """生成运行时类配置"""
        runtime_configs = {
            'gVisor': {
                'runtimeClassName': 'gvisor',
                'handler': 'gvisor',
                'description': 'Google gVisor沙箱运行时'
            },
            'KataContainers': {
                'runtimeClassName': 'kata',
                'handler': 'kata',
                'description': 'Kata Containers轻量级虚拟机'
            }
        }
        
        if runtime_name not in runtime_configs:
            raise ValueError(f"Unsupported runtime: {runtime_name}")
        
        config = runtime_configs[runtime_name]
        
        return {
            'apiVersion': 'node.k8s.io/v1beta1',
            'kind': 'RuntimeClass',
            'metadata': {
                'name': config['runtimeClassName']
            },
            'handler': config['handler'],
            'scheduling': {
                'nodeSelector': {
                    'runtime': runtime_name
                }
            }
        }
    
    def configure_app_armor_profile(self, profile_name: str, app_name: str) -> Dict[str, Any]:
        """配置AppArmor安全策略"""
        return {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': app_name,
                'annotations': {
                    'container.apparmor.security.beta.kubernetes.io/' + app_name: profile_name
                }
            }
        }

# 演示Pod安全配置
def demonstrate_pod_security():
    """演示Pod安全配置"""
    print("=== Kubernetes Pod安全配置演示 ===")
    
    pod_security_mgr = PodSecurityManager()
    runtime_security = RuntimeSecurity()
    
    # 生成不同安全级别的Pod配置
    security_levels = [PodSecurityLevel.BASELINE, PodSecurityLevel.RESTRICTED]
    
    for level in security_levels:
        print(f"\n{level.value.upper()} 级别Pod配置:")
        
        pod_spec = pod_security_mgr.generate_secure_pod_spec(
            app_name='secure-app',
            image='nginx:1.19',
            security_level=level,
            run_as_user=1000
        )
        
        print(f"  Pod名称: {pod_spec['metadata']['name']}")
        print(f"  安全级别: {pod_spec['metadata']['labels']['security-level']}")
        
        # 显示安全上下文配置
        security_context = pod_spec['spec']['securityContext']
        container_security = pod_spec['spec']['containers'][0]['securityContext']
        
        print(f"  非root运行: {security_context['runAsNonRoot']}")
        print(f"  禁用权限提升: {container_security['allowPrivilegeEscalation']}")
        print(f"  只读根文件系统: {container_security['readOnlyRootFilesystem']}")
        print(f"  能力控制: {container_security['capabilities']['drop']}")
        
        # 安全合规性检查
        compliance = pod_security_mgr.validate_security_compliance(pod_spec)
        print(f"  合规状态: {'通过' if compliance['compliant'] else '不通过'}")
        
        if compliance['violations']:
            for violation in compliance['violations']:
                print(f"    违规: {violation['issue']}")
    
    # 运行时安全配置
    print("\n容器运行时安全:")
    
    # gVisor运行时
    gvisor_config = runtime_security.generate_runtime_class_config('gVisor')
    print(f"  gVisor运行时类: {gvisor_config['metadata']['name']}")
    print(f"  处理程序: {gvisor_config['handler']}")
    
    # Kata Containers运行时
    kata_config = runtime_security.generate_runtime_class_config('KataContainers')
    print(f"  Kata运行时类: {kata_config['metadata']['name']}")
    print(f"  处理程序: {kata_config['handler']}")
    
    # 安全建议
    print("\n安全最佳实践建议:")
    best_practices = [
        "始终使用非特权用户运行容器",
        "限制容器权限和能力",
        "使用只读根文件系统",
        "定期更新基础镜像",
        "实施Pod安全策略",
        "使用运行时沙箱隔离",
        "配置资源限制",
        "启用审计日志",
        "监控异常行为",
        "实施零信任网络"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")

# 运行演示
demonstrate_pod_security()
```

## 镜像安全管理

### 镜像安全生命周期

```mermaid
graph LR
    A[镜像构建] --> B[镜像扫描]
    B --> C[镜像签名]
    C --> D[镜像存储]
    D --> E[镜像分发]
    E --> F[镜像部署]
    F --> G[运行时监控]
    
    B --> H[漏洞检测]
    C --> I[完整性验证]
    D --> J[访问控制]
    F --> K[策略执行]
    G --> L[行为监控]
    
    H --> M[修复更新]
    I --> M
    J --> M
    K --> M
    L --> M
```

### 镜像安全扫描与验证

```python
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class VulnerabilitySeverity(Enum):
    """漏洞严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class TrustLevel(Enum):
    """镜像信任级别"""
    VERIFIED = "verified"
    SIGNED = "signed"
    UNVERIFIED = "unverified"
    UNTRUSTED = "untrusted"

@dataclass
class VulnerabilityInfo:
    """漏洞信息"""
    cve_id: str
    severity: VulnerabilitySeverity
    package_name: str
    package_version: str
    fixed_version: Optional[str]
    description: str
    cvss_score: float

@dataclass
class ImageSecurityResult:
    """镜像安全扫描结果"""
    image_name: str
    scan_timestamp: str
    vulnerabilities: List[VulnerabilityInfo]
    trust_level: TrustLevel
    compliance_status: bool
    recommendations: List[str]

class ImageSecurityManager:
    """镜像安全管理器"""
    
    def __init__(self):
        self.vulnerability_database = self._initialize_vulnerability_db()
        self.trusted_registries = {}
        self.scan_policies = {}
        self.signature_verifiers = {}
    
    def _initialize_vulnerability_db(self) -> Dict[str, List[VulnerabilityInfo]]:
        """初始化漏洞数据库"""
        return {
            'nginx:1.19': [
                VulnerabilityInfo(
                    cve_id='CVE-2021-36158',
                    severity=VulnerabilitySeverity.HIGH,
                    package_name='libcrypt',
                    package_version='1.8.5',
                    fixed_version='1.8.6',
                    description='OpenSSL 错误修复中的漏洞',
                    cvss_score=7.5
                ),
                VulnerabilityInfo(
                    cve_id='CVE-2021-3449',
                    severity=VulnerabilitySeverity.MEDIUM,
                    package_name='openssl',
                    package_version='1.1.1k',
                    fixed_version='1.1.1k-1',
                    description='OpenSSL NULL指针取消引用',
                    cvss_score=5.9
                )
            ],
            'alpine:3.14': [
                VulnerabilityInfo(
                    cve_id='CVE-2021-36159',
                    severity=VulnerabilitySeverity.HIGH,
                    package_name='musl',
                    package_version='1.2.2',
                    fixed_version='1.2.3',
                    description='musl libc 缓冲区溢出',
                    cvss_score=8.1
                )
            ]
        }
    
    def scan_image_vulnerabilities(self, image_name: str, 
                                  trusted_registries: Optional[List[str]] = None) -> ImageSecurityResult:
        """扫描镜像漏洞"""
        print(f"正在扫描镜像: {image_name}")
        
        # 模拟漏洞扫描过程
        vulnerabilities = self.vulnerability_database.get(image_name, [])
        
        # 计算漏洞统计
        vuln_stats = {
            VulnerabilitySeverity.CRITICAL: 0,
            VulnerabilitySeverity.HIGH: 0,
            VulnerabilitySeverity.MEDIUM: 0,
            VulnerabilitySeverity.LOW: 0,
            VulnerabilitySeverity.UNKNOWN: 0
        }
        
        for vuln in vulnerabilities:
            vuln_stats[vuln.severity] += 1
        
        # 评估信任级别
        trust_level = self._evaluate_trust_level(image_name, trusted_registries)
        
        # 检查合规性
        compliance_status = self._check_compliance(vulnerabilities)
        
        # 生成建议
        recommendations = self._generate_recommendations(vulnerabilities, trust_level)
        
        return ImageSecurityResult(
            image_name=image_name,
            scan_timestamp=datetime.now().isoformat(),
            vulnerabilities=vulnerabilities,
            trust_level=trust_level,
            compliance_status=compliance_status,
            recommendations=recommendations
        )
    
    def _evaluate_trust_level(self, image_name: str, 
                             trusted_registries: Optional[List[str]] = None) -> TrustLevel:
        """评估镜像信任级别"""
        if not trusted_registries:
            trusted_registries = ['docker.io', 'registry.k8s.io', 'gcr.io']
        
        # 检查镜像来源
        registry = image_name.split('/')[0] if '/' in image_name else 'docker.io'
        
        if registry in trusted_registries:
            # 检查是否有签名
            if self._verify_image_signature(image_name):
                return TrustLevel.VERIFIED
            else:
                return TrustLevel.SIGNED
        else:
            return TrustLevel.UNTRUSTED
    
    def _verify_image_signature(self, image_name: str) -> bool:
        """验证镜像签名"""
        # 模拟签名验证过程
        # 实际环境中会使用Notary或其他签名验证工具
        return True  # 假设签名验证通过
    
    def _check_compliance(self, vulnerabilities: List[VulnerabilityInfo]) -> bool:
        """检查合规性"""
        # 检查是否存在高危或严重漏洞
        critical_or_high = any(
            vuln.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]
            for vuln in vulnerabilities
        )
        
        return not critical_or_high
    
    def _generate_recommendations(self, vulnerabilities: List[VulnerabilityInfo], 
                                 trust_level: TrustLevel) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 基于漏洞的建议
        if vulnerabilities:
            recommendations.append("更新到修复了已知漏洞的镜像版本")
        
        critical_vulns = [v for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]
        if critical_vulns:
            recommendations.append(f"立即修复 {len(critical_vulns)} 个严重漏洞")
        
        high_vulns = [v for v in vulnerabilities if v.severity == VulnerabilitySeverity.HIGH]
        if high_vulns:
            recommendations.append(f"优先处理 {len(high_vulns)} 个高危漏洞")
        
        # 基于信任级别的建议
        if trust_level == TrustLevel.UNTRUSTED:
            recommendations.append("仅使用来自可信镜像仓库的镜像")
        elif trust_level == TrustLevel.UNVERIFIED:
            recommendations.append("验证镜像签名以确保完整性")
        
        # 通用建议
        recommendations.extend([
            "实施镜像扫描作为CI/CD流水线的一部分",
            "使用镜像版本标签而不是'latest'",
            "定期更新基础镜像",
            "监控镜像安全公告",
            "实施供应链安全策略"
        ])
        
        return recommendations
    
    def create_image_policy(self, namespace: str, 
                           allowed_images: List[str],
                           trusted_registries: List[str]) -> Dict[str, Any]:
        """创建镜像策略"""
        return {
            'apiVersion': 'admissionregistration.k8s.io/v1',
            'kind': 'ValidatingAdmissionPolicy',
            'metadata': {
                'name': f'image-security-policy-{namespace}'
            },
            'spec': {
                'matchConstraints': {
                    'resourceRules': [{
                        'apiGroups': ['', 'apps'],
                        'apiVersions': ['v1', 'apps/v1'],
                        'operations': ['CREATE', 'UPDATE'],
                        'resources': ['pods', 'deployments', 'statefulsets', 'daemonsets']
                    }]
                },
                'validations': [
                    # 允许的镜像检查
                    {
                        'expression': 'image in ' + str(allowed_images),
                        'message': 'Only images from approved list are allowed'
                    },
                    # 镜像标签检查
                    {
                        'expression': '!image.contains(":latest")',
                        'message': 'Using ":latest" tag is not allowed'
                    },
                    # 镜像仓库检查
                    {
                        'expression': 'split(image, "/")[0] in ' + str(trusted_registries),
                        'message': 'Only images from trusted registries are allowed'
                    },
                    # 镜像漏洞检查（在实际环境中由外部工具实现）
                    {
                        'expression': 'imageScanned == true',
                        'message': 'Images must be security scanned before deployment'
                    }
                ]
            }
        }
    
    def generate_signature_verification_config(self) -> Dict[str, Any]:
        """生成签名验证配置"""
        return {
            'apiVersion': 'config.gatekeeper.sh/v1alpha1',
            'kind': 'Config',
            'metadata': {
                'name': 'config'
            },
            'spec': {
                'match': [{
                    'versions': ['v1'],
                    'kinds': ['Pod', 'Deployment', 'StatefulSet']
                }],
                'parameters': {
                    'verification': {
                        'required': True,
                        'trust_roots': [
                            'docker.io/docker.io/library',
                            'registry.k8s.io/kubernetesio'
                        ]
                    }
                }
            }
        }

class SupplyChainSecurity:
    """供应链安全"""
    
    def __init__(self):
        self.sbom_generator = SBOMGenerator()
        self.provenance_tracker = ProvenanceTracker()
    
    def create_sbom_policy(self) -> Dict[str, Any]:
        """创建软件物料清单(SBOM)策略"""
        return {
            'apiVersion': 'admissionregistration.k8s.io/v1',
            'kind': 'MutatingAdmissionPolicy',
            'metadata': {
                'name': 'require-sbom'
            },
            'spec': {
                'matchConstraints': {
                    'resourceRules': [{
                        'apiGroups': ['', 'apps'],
                        'apiVersions': ['v1', 'apps/v1'],
                        'operations': ['CREATE', 'UPDATE'],
                        'resources': ['pods', 'deployments']
                    }]
                },
                'mutations': [
                    {
                        'expression': 'addAnnotation("sbom.generated", "true")',
                        'description': '添加SBOM生成标记'
                    }
                ],
                'validations': [
                    {
                        'expression': 'hasAnnotation("sbom.generated")',
                        'message': 'SBOM must be generated for all images'
                    }
                ]
            }
        }

class SBOMGenerator:
    """SBOM生成器"""
    
    def generate_sbom(self, image_name: str) -> Dict[str, Any]:
        """生成软件物料清单"""
        # 模拟SBOM生成过程
        return {
            'bomFormat': 'CycloneDX',
            'specVersion': '1.4',
            'serialNumber': 'urn:uuid:' + hashlib.md5(image_name.encode()).hexdigest(),
            'version': 1,
            'metadata': {
                'component': {
                    'bom-ref': image_name,
                    'type': 'container',
                    'name': image_name,
                    'version': 'latest'
                }
            },
            'components': [
                {
                    'bom-ref': 'openssl',
                    'type': 'library',
                    'name': 'openssl',
                    'version': '1.1.1k'
                },
                {
                    'bom-ref': 'nginx',
                    'type': 'application',
                    'name': 'nginx',
                    'version': '1.19'
                }
            ],
            'dependencies': [
                {
                    'ref': 'nginx',
                    'dependsOn': ['openssl']
                }
            ]
        }

class ProvenanceTracker:
    """来源追踪器"""
    
    def track_image_provenance(self, image_name: str, 
                              build_info: Dict[str, Any]) -> Dict[str, Any]:
        """追踪镜像来源"""
        return {
            'image': image_name,
            'build_info': build_info,
            'provenance': {
                'builder': build_info.get('builder', 'unknown'),
                'build_date': build_info.get('build_date'),
                'git_commit': build_info.get('git_commit'),
                'dockerfile_hash': build_info.get('dockerfile_hash'),
                'source_checksum': build_info.get('source_checksum')
            },
            'attestations': [
                {
                    'type': 'provenance',
                    'predicate': {
                        'buildType': 'buildkit@1',
                        'builder': build_info.get('builder')
                    }
                }
            ]
        }

# 演示镜像安全
def demonstrate_image_security():
    """演示镜像安全管理"""
    print("=== Kubernetes镜像安全管理演示 ===")
    
    image_security_mgr = ImageSecurityManager()
    supply_chain_security = SupplyChainSecurity()
    
    # 扫描不同镜像
    images_to_scan = [
        'nginx:1.19',
        'alpine:3.14',
        'ubuntu:20.04'
    ]
    
    trusted_registries = ['docker.io', 'registry.k8s.io']
    
    for image in images_to_scan:
        print(f"\n--- 扫描镜像: {image} ---")
        
        scan_result = image_security_mgr.scan_image_vulnerabilities(
            image, trusted_registries
        )
        
        print(f"扫描时间: {scan_result.scan_timestamp}")
        print(f"信任级别: {scan_result.trust_level.value}")
        print(f"合规状态: {'合规' if scan_result.compliance_status else '不合规'}")
        
        # 漏洞统计
        vuln_counts = {}
        for vuln in scan_result.vulnerabilities:
            severity = vuln.severity.value
            vuln_counts[severity] = vuln_counts.get(severity, 0) + 1
        
        print(f"漏洞统计:")
        for severity, count in vuln_counts.items():
            print(f"  {severity}: {count}")
        
        if scan_result.vulnerabilities:
            print("主要漏洞:")
            for vuln in scan_result.vulnerabilities[:3]:  # 显示前3个漏洞
                print(f"  {vuln.cve_id} ({vuln.severity.value}) - {vuln.description}")
        
        # 安全建议
        print("安全建议:")
        for recommendation in scan_result.recommendations[:3]:
            print(f"  • {recommendation}")
    
    # 镜像策略
    print("\n--- 镜像安全策略配置 ---")
    
    allowed_images = ['nginx:1.19', 'alpine:3.14', 'redis:6.2']
    policy = image_security_mgr.create_image_policy(
        'production', allowed_images, trusted_registries
    )
    
    print(f"策略名称: {policy['metadata']['name']}")
    print(f"验证规则数量: {len(policy['spec']['validations'])}")
    
    for i, validation in enumerate(policy['spec']['validations'], 1):
        print(f"  {i}. {validation['message']}")
    
    # 签名验证配置
    signature_config = image_security_mgr.generate_signature_verification_config()
    print(f"\n签名验证配置: {signature_config['metadata']['name']}")
    
    # SBOM和供应链安全
    print("\n--- 供应链安全 ---")
    
    sbom_policy = supply_chain_security.create_sbom_policy()
    print(f"SBOM策略: {sbom_policy['metadata']['name']}")
    
    # 生成SBOM
    sbom = supply_chain_security.sbom_generator.generate_sbom('nginx:1.19')
    print(f"SBOM格式: {sbom['bomFormat']}")
    print(f"组件数量: {len(sbom['components'])}")
    
    # 来源追踪
    build_info = {
        'builder': 'docker-buildx',
        'build_date': datetime.now().isoformat(),
        'git_commit': 'abc123def456',
        'dockerfile_hash': hashlib.md5(b'FROM nginx:1.19').hexdigest()
    }
    
    provenance = supply_chain_security.provenance_tracker.track_image_provenance(
        'nginx:1.19', build_info
    )
    
    print(f"构建器: {provenance['provenance']['builder']}")
    print(f"构建日期: {provenance['provenance']['build_date']}")
    
    print("\n镜像安全最佳实践:")
    best_practices = [
        "仅使用来自可信镜像仓库的镜像",
        "实施镜像签名和验证",
        "在CI/CD流水线中集成镜像扫描",
        "定期更新基础镜像",
        "使用具体的版本标签而非'latest'",
        "监控镜像安全公告",
        "实施软件物料清单(SBOM)",
        "验证镜像构建来源",
        "定期进行漏洞评估",
        "建立镜像安全响应流程"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")

# 运行演示
demonstrate_image_security()
```

## 数据加密与密钥管理

### 加密架构原理

```mermaid
graph TB
    subgraph "静态数据加密"
        A[etcd数据] --> B[EncryptionProvider]
        C[Secret对象] --> B
        D[ConfigMap] --> B
        B --> E[AES-GCM加密]
        B --> F[key轮换机制]
    end
    
    subgraph "传输数据加密"
        G[API Server通信] --> H[TLS加密]
        I[Pod间通信] --> H
        J[外部通信] --> H
        H --> K[证书管理]
        H --> L[证书轮换]
    end
    
    subgraph "密钥管理"
        M[KMS提供商] --> N[密钥存储]
        O[外部密钥管理] --> N
        P[HSM硬件安全模块] --> N
        N --> Q[密钥生命周期]
    end
    
    B --> M
    H --> K
    K --> M
```

### 加密实现详解

```python
import base64
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class EncryptionProvider(Enum):
    """加密提供者类型"""
    AES_GCM = "aescbc"
    AES_CBC = "aescbc"
    SECRETBOX = "secretbox"
    KMS = "kms"

class KeyRotationPolicy(Enum):
    """密钥轮换策略"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"

@dataclass
class EncryptionKey:
    """加密密钥信息"""
    key_id: str
    key_data: str
    created_at: str
    last_used: Optional[str] = None
    status: str = "active"

class EncryptionManager:
    """加密管理器"""
    
    def __init__(self):
        self.encryption_config = {}
        self.keys = {}
        self.kms_providers = {}
        self._initialize_encryption_config()
    
    def _initialize_encryption_config(self):
        """初始化加密配置"""
        # 生成示例密钥
        for i in range(2):
            key_id = f"key-{i+1}"
            key_data = base64.b64encode(os.urandom(32)).decode('utf-8')
            self.keys[key_id] = EncryptionKey(
                key_id=key_id,
                key_data=key_data,
                created_at=datetime.now().isoformat()
            )
    
    def create_encryption_configuration(self, 
                                       resources: List[str],
                                       provider: EncryptionProvider = EncryptionProvider.AES_GCM) -> Dict[str, Any]:
        """创建加密配置"""
        
        if provider == EncryptionProvider.AES_GCM:
            providers = [
                {
                    "identity": {}
                },
                {
                    "aescbc": {
                        "keys": [
                            {
                                "name": key_id,
                                "secret": key_data
                            }
                            for key_id, key_data in [(key.key_id, key.key_data) 
                                                   for key in self.keys.values()]
                        ]
                    }
                }
            ]
        
        config = {
            "apiVersion": "apiserver.config.k8s.io/v1",
            "kind": "EncryptionConfiguration",
            "resources": [
                {
                    "resources": resources,
                    "providers": providers
                }
            ]
        }
        
        self.encryption_config["config"] = config
        return config
    
    def generate_kms_configuration(self, kms_provider: str = "azure") -> Dict[str, Any]:
        """生成KMS配置"""
        
        if kms_provider == "azure":
            kms_config = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "encryption-config",
                    "namespace": "kube-system"
                },
                "stringData": {
                    "encryption-config.yaml": f"""
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
      - configmaps
    providers:
      - kms:
          name: azure-kms
          timeout: 3s
          cacheSize: 100
          endpoint: unix:///tmp/kms/azurekms.sock
"""
                }
            }
        else:
            kms_config = {
                "apiVersion": "apiserver.config.k8s.io/v1",
                "kind": "EncryptionConfiguration",
                "resources": [
                    {
                        "resources": ["secrets", "configmaps"],
                        "providers": [
                            {
                                "kms": {
                                    "name": "default-kms",
                                    "timeout": 3,
                                    "cacheSize": 10,
                                    "endpoint": "/tmp/kms/kms.sock"
                                }
                            }
                        ]
                    }
                ]
            }
        
        return kms_config
    
    def rotate_encryption_key(self, key_id: str, policy: KeyRotationPolicy = KeyRotationPolicy.AUTOMATIC) -> bool:
        """执行密钥轮换"""
        try:
            # 查找要轮换的密钥
            old_key = self.keys.get(key_id)
            if not old_key:
                print(f"密钥 {key_id} 不存在")
                return False
            
            # 生成新密钥
            new_key_id = f"key-{len(self.keys) + 1}"
            new_key_data = base64.b64encode(os.urandom(32)).decode('utf-8')
            new_key = EncryptionKey(
                key_id=new_key_id,
                key_data=new_key_data,
                created_at=datetime.now().isoformat()
            )
            
            # 添加新密钥
            self.keys[new_key_id] = new_key
            old_key.status = "rotated"
            old_key.last_used = datetime.now().isoformat()
            
            print(f"密钥轮换完成: {key_id} -> {new_key_id}")
            return True
            
        except Exception as e:
            print(f"密钥轮换失败: {e}")
            return False
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """获取加密状态"""
        active_keys = [key for key in self.keys.values() if key.status == "active"]
        rotated_keys = [key for key in self.keys.values() if key.status == "rotated"]
        
        return {
            "total_keys": len(self.keys),
            "active_keys": len(active_keys),
            "rotated_keys": len(rotated_keys),
            "has_encryption_config": "config" in self.encryption_config,
            "encryption_enabled": len(active_keys) > 0
        }

class CertificateManager:
    """证书管理器"""
    
    def __init__(self):
        self.certificates = {}
        self.ca_config = {}
        self._initialize_certificate_authority()
    
    def _initialize_certificate_authority(self):
        """初始化证书颁发机构"""
        # 生成CA证书配置
        self.ca_config = {
            "ca_cert": "-----BEGIN CERTIFICATE-----\n...",
            "ca_key": "-----BEGIN PRIVATE KEY-----\n...",
            "ca_path": "/etc/kubernetes/pki/ca.crt"
        }
    
    def generate_server_certificate(self, 
                                  common_name: str,
                                  sans: List[str] = None) -> Dict[str, Any]:
        """生成服务器证书"""
        
        if sans is None:
            sans = []
        
        cert_info = {
            "cn": common_name,
            "sans": sans,
            "cert_type": "server",
            "validity_period": "8760h",  # 1年
            "key_usage": ["keyEncipherment", "serverAuth"],
            "extended_key_usage": ["serverAuth"]
        }
        
        self.certificates[common_name] = cert_info
        return cert_info
    
    def generate_client_certificate(self, 
                                  common_name: str,
                                  organizations: List[str] = None) -> Dict[str, Any]:
        """生成客户端证书"""
        
        if organizations is None:
            organizations = []
        
        cert_info = {
            "cn": common_name,
            "organizations": organizations,
            "cert_type": "client",
            "validity_period": "8760h",  # 1年
            "key_usage": ["digitalSignature", "clientAuth"],
            "extended_key_usage": ["clientAuth"]
        }
        
        self.certificates[common_name] = cert_info
        return cert_info
    
    def create_certificate_request(self, cert_info: Dict[str, Any]) -> Dict[str, Any]:
        """创建证书签名请求"""
        
        csr_config = {
            "apiVersion": "certificates.k8s.io/v1",
            "kind": "CertificateSigningRequest",
            "metadata": {
                "name": f"csr-{cert_info['cn']}"
            },
            "spec": {
                "request": base64.b64encode(f"""
-----BEGIN CERTIFICATE REQUEST-----
{base64.b64encode(f"CN={cert_info['cn']}".encode()).decode()}
-----END CERTIFICATE REQUEST-----
""".encode()).decode(),
                "signerName": "kubernetes.io/kube-apiserver-client",
                "expirationSeconds": 3153600000  # 100年
            }
        }
        
        return csr_config
    
    def monitor_certificate_expiry(self) -> Dict[str, List[Dict[str, Any]]]:
        """监控证书过期"""
        
        expired_certs = []
        expiring_soon = []
        valid_certs = []
        
        for cert_name, cert_info in self.certificates.items():
            # 模拟证书过期时间检查
            cert_status = {
                "cert_name": cert_name,
                "type": cert_info["cert_type"],
                "days_until_expiry": 365  # 模拟值
            }
            
            if cert_status["days_until_expiry"] < 0:
                expired_certs.append(cert_status)
            elif cert_status["days_until_expiry"] < 30:
                expiring_soon.append(cert_status)
            else:
                valid_certs.append(cert_status)
        
        return {
            "expired": expired_certs,
            "expiring_soon": expiring_soon,
            "valid": valid_certs
        }

def demonstrate_encryption_and_certificates():
    """演示加密和证书管理"""
    print("=" * 60)
    print("Kubernetes 数据加密与证书管理演示")
    print("=" * 60)
    
    # 创建加密管理器
    encryption_mgr = EncryptionManager()
    
    # 创建加密配置
    print("\n--- 创建加密配置 ---")
    resources = ["secrets", "configmaps"]
    config = encryption_mgr.create_encryption_configuration(
        resources, EncryptionProvider.AES_GCM
    )
    
    print(f"加密资源配置: {config['apiVersion']}/{config['kind']}")
    print(f"资源类型: {resources}")
    print(f"加密提供者数量: {len(config['resources'][0]['providers'])}")
    
    # KMS配置
    print("\n--- KMS集成配置 ---")
    kms_config = encryption_mgr.generate_kms_configuration("azure")
    print(f"KMS配置类型: {kms_config['kind']}")
    
    # 密钥轮换
    print("\n--- 密钥轮换演示 ---")
    
    # 显示当前密钥状态
    status = encryption_mgr.get_encryption_status()
    print(f"加密状态:")
    print(f"  总密钥数: {status['total_keys']}")
    print(f"  活跃密钥: {status['active_keys']}")
    print(f"  已轮换密钥: {status['rotated_keys']}")
    
    # 执行密钥轮换
    if "key-1" in encryption_mgr.keys:
        success = encryption_mgr.rotate_encryption_key("key-1")
        print(f"密钥轮换成功: {success}")
        
        # 显示轮换后的状态
        new_status = encryption_mgr.get_encryption_status()
        print(f"轮换后状态: {new_status}")
    
    # 创建证书管理器
    print("\n--- 证书管理演示 ---")
    cert_mgr = CertificateManager()
    
    # 生成服务器证书
    server_cert = cert_mgr.generate_server_certificate(
        "kube-apiserver",
        ["localhost", "127.0.0.1", "*.cluster.local"]
    )
    print(f"服务器证书生成:")
    print(f"  主题名称: {server_cert['cn']}")
    print(f"  SANs: {server_cert['sans']}")
    print(f"  证书类型: {server_cert['cert_type']}")
    
    # 生成客户端证书
    client_cert = cert_mgr.generate_client_certificate(
        "admin-user",
        ["system:masters"]
    )
    print(f"\n客户端证书生成:")
    print(f"  主题名称: {client_cert['cn']}")
    print(f"  组织: {client_cert['organizations']}")
    print(f"  证书类型: {client_cert['cert_type']}")
    
    # 创建证书签名请求
    print("\n--- 证书签名请求 ---")
    csr = cert_mgr.create_certificate_request(server_cert)
    print(f"CSR配置: {csr['apiVersion']}/{csr['kind']}")
    print(f"CSR名称: {csr['metadata']['name']}")
    print(f"签名者: {csr['spec']['signerName']}")
    
    # 证书过期监控
    print("\n--- 证书过期监控 ---")
    expiry_status = cert_mgr.monitor_certificate_expiry()
    print(f"有效证书: {len(expiry_status['valid'])}")
    print(f"即将过期证书: {len(expiry_status['expiring_soon'])}")
    print(f"已过期证书: {len(expiry_status['expired'])}")
    
    print("\n数据加密与证书管理最佳实践:")
    best_practices = [
        "启用etcd静态数据加密",
        "使用AES-GCM或更高级的加密算法",
        "定期轮换加密密钥",
        "集成外部KMS服务",
        "使用自签名证书或企业CA",
        "定期更新证书和密钥",
        "监控证书过期时间",
        "启用mTLS进行服务间通信",
        "实施证书透明度监控",
        "建立证书泄露应急响应流程"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")

# 运行演示
demonstrate_encryption_and_certificates()
```

## 安全监控与合规性管理

### 监控架构原理

```mermaid
graph TB
    subgraph "安全事件收集"
        A[API审计日志] --> B[事件聚合器]
        C[Pod安全事件] --> B
        D[网络流量] --> B
        E[系统调用] --> B
    end
    
    subgraph "实时分析"
        B --> F[威胁检测引擎]
        F --> G[异常行为分析]
        G --> H[模式识别]
        H --> I[关联分析]
    end
    
    subgraph "告警与响应"
        I --> J[安全告警]
        J --> K[自动响应]
        K --> L[事件记录]
        K --> M[合规报告]
    end
    
    subgraph "合规性检查"
        N[CIS基准] --> O[配置扫描]
        P[NIST框架] --> O
        Q[GDPR要求] --> O
        O --> R[合规报告]
    end
```

### 监控实现详解

```python
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib

class Severity(Enum):
    """安全事件严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class EventType(Enum):
    """安全事件类型"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALWARE_DETECTED = "malware_detected"
    DATA_EXFILTRATION = "data_exfiltration"

class ComplianceStandard(Enum):
    """合规性标准"""
    CIS_BENCHMARK = "cis_benchmark"
    NIST_FRAMEWORK = "nist_framework"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"

@dataclass
class SecurityEvent:
    """安全事件"""
    event_id: str
    event_type: EventType
    severity: Severity
    source: str
    target: str
    description: str
    timestamp: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = hashlib.md5(
                f"{self.timestamp}{self.event_type.value}{self.source}".encode()
            ).hexdigest()

@dataclass
class ComplianceCheck:
    """合规性检查结果"""
    check_id: str
    standard: ComplianceStandard
    control_id: str
    control_name: str
    status: str  # "pass", "fail", "warning", "not_applicable"
    description: str
    severity: Severity
    timestamp: str

class SecurityMonitor:
    """安全监控器"""
    
    def __init__(self):
        self.events = []
        self.event_patterns = {}
        self.threat_intelligence = {}
        self.anomaly_detectors = {}
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """初始化监控配置"""
        # 设置事件模式
        self.event_patterns = {
            "brute_force": {
                "pattern": ["authentication_failure"] * 5,
                "time_window": 300,  # 5分钟
                "threshold": 5
            },
            "privilege_escalation": {
                "pattern": ["unauthorized_access", "policy_violation"],
                "time_window": 600,
                "threshold": 1
            },
            "data_exfiltration": {
                "pattern": ["data_exfiltration"],
                "time_window": 3600,
                "threshold": 1
            }
        }
    
    def collect_security_event(self, 
                             event_type: EventType,
                             severity: Severity,
                             source: str,
                             target: str,
                             description: str,
                             metadata: Dict[str, Any] = None) -> str:
        """收集安全事件"""
        
        if metadata is None:
            metadata = {}
        
        event = SecurityEvent(
            event_id="",
            event_type=event_type,
            severity=severity,
            source=source,
            target=target,
            description=description,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self.events.append(event)
        print(f"安全事件已收集: {event.event_id} - {event_type.value}")
        
        # 执行实时威胁检测
        self._detect_threats(event)
        
        return event.event_id
    
    def _detect_threats(self, event: SecurityEvent):
        """威胁检测"""
        recent_events = self._get_recent_events(3600)  # 最近1小时
        
        # 检查威胁模式
        for pattern_name, pattern_config in self.event_patterns.items():
            if self._matches_pattern(recent_events, pattern_config):
                self._trigger_threat_alert(pattern_name, event)
    
    def _get_recent_events(self, time_window: int) -> List[SecurityEvent]:
        """获取最近的事件"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        return [
            event for event in self.events
            if datetime.fromisoformat(event.timestamp) > cutoff_time
        ]
    
    def _matches_pattern(self, events: List[SecurityEvent], pattern_config: Dict[str, Any]) -> bool:
        """检查事件模式匹配"""
        pattern = pattern_config["pattern"]
        threshold = pattern_config["threshold"]
        time_window = pattern_config["time_window"]
        
        # 简化的模式匹配逻辑
        event_types = [event.event_type.value for event in events[-threshold:]]
        
        return event_types == pattern[:len(event_types)]
    
    def _trigger_threat_alert(self, pattern_name: str, event: SecurityEvent):
        """触发威胁告警"""
        alert = {
            "alert_id": f"alert-{int(time.time())}",
            "pattern": pattern_name,
            "severity": event.severity.value,
            "triggering_event": event.event_id,
            "timestamp": datetime.now().isoformat(),
            "description": f"检测到{pattern_name}威胁模式"
        }
        
        print(f"🚨 安全告警: {alert['alert_id']} - {alert['description']}")
        
        # 这里可以集成告警系统 (如PagerDuty, Slack, Email等)
        self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """发送告警"""
        # 模拟告警发送
        alert_channels = ["email", "slack", "pagerduty"]
        print(f"告警已发送到: {alert_channels}")
    
    def analyze_threat_trends(self) -> Dict[str, Any]:
        """威胁趋势分析"""
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        hourly_activity = defaultdict(int)
        
        for event in self.events:
            event_counts[event.event_type.value] += 1
            severity_counts[event.severity.value] += 1
            
            # 按小时统计活动
            hour = datetime.fromisoformat(event.timestamp).hour
            hourly_activity[hour] += 1
        
        return {
            "total_events": len(self.events),
            "event_type_distribution": dict(event_counts),
            "severity_distribution": dict(severity_counts),
            "hourly_activity": dict(hourly_activity),
            "analysis_timestamp": datetime.now().isoformat()
        }

class ComplianceManager:
    """合规性管理器"""
    
    def __init__(self):
        self.compliance_checks = []
        self.compliance_standards = {}
        self._initialize_compliance_standards()
    
    def _initialize_compliance_standards(self):
        """初始化合规性标准"""
        self.compliance_standards = {
            ComplianceStandard.CIS_BENCHMARK: [
                {
                    "control_id": "1.1.1",
                    "control_name": "确保Master节点配置文件权限",
                    "description": "Master节点配置文件应该只能被root用户访问",
                    "severity": Severity.HIGH
                },
                {
                    "control_id": "1.2.1",
                    "control_name": "确保API Server使用匿名认证",
                    "description": "应该禁用匿名认证或严格控制匿名访问",
                    "severity": Severity.MEDIUM
                },
                {
                    "control_id": "1.3.1",
                    "control_name": "确保etcd数据目录权限",
                    "description": "etcd数据目录应该只能被etcd用户访问",
                    "severity": Severity.HIGH
                }
            ],
            ComplianceStandard.NIST_FRAMEWORK: [
                {
                    "control_id": "AC-2",
                    "control_name": "Account Management",
                    "description": "应该建立、管理和监控用户账户",
                    "severity": Severity.HIGH
                },
                {
                    "control_id": "SC-7",
                    "control_name": "Boundary Protection",
                    "description": "应该实施边界保护措施",
                    "severity": Severity.MEDIUM
                },
                {
                    "control_id": "SI-4",
                    "control_name": "System Monitoring",
                    "description": "应该实施系统监控",
                    "severity": Severity.HIGH
                }
            ]
        }
    
    def run_compliance_check(self, standard: ComplianceStandard) -> List[ComplianceCheck]:
        """运行合规性检查"""
        
        checks = []
        control_list = self.compliance_standards.get(standard, [])
        
        for control in control_list:
            check = ComplianceCheck(
                check_id=f"{control['control_id']}-{int(time.time())}",
                standard=standard,
                control_id=control["control_id"],
                control_name=control["control_name"],
                status=self._simulate_check_result(control["severity"]),
                description=control["description"],
                severity=control["severity"],
                timestamp=datetime.now().isoformat()
            )
            checks.append(check)
            self.compliance_checks.append(check)
        
        return checks
    
    def _simulate_check_result(self, severity: Severity) -> str:
        """模拟检查结果 (实际应该执行真实的合规性检查)"""
        import random
        
        # 模拟不同的通过率，高危项通过率较低
        if severity == Severity.CRITICAL:
            weights = [0.7, 0.2, 0.1]  # 70%通过, 20%失败, 10%警告
        elif severity == Severity.HIGH:
            weights = [0.8, 0.15, 0.05]
        else:
            weights = [0.9, 0.05, 0.05]
        
        return random.choices(["pass", "fail", "warning"], weights=weights)[0]
    
    def generate_compliance_report(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """生成合规性报告"""
        
        # 获取指定标准的检查结果
        standard_checks = [
            check for check in self.compliance_checks
            if check.standard == standard
        ]
        
        if not standard_checks:
            return {"error": f"No compliance checks found for {standard.value}"}
        
        # 统计结果
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for check in standard_checks:
            status_counts[check.status] += 1
            severity_counts[check.severity.value] += 1
        
        # 计算合规性分数
        total_checks = len(standard_checks)
        passed_checks = status_counts["pass"]
        compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        report = {
            "standard": standard.value,
            "report_timestamp": datetime.now().isoformat(),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": status_counts["fail"],
            "warning_checks": status_counts["warning"],
            "compliance_score": round(compliance_score, 2),
            "status_distribution": dict(status_counts),
            "severity_distribution": dict(severity_counts),
            "failed_controls": [
                {
                    "control_id": check.control_id,
                    "control_name": check.control_name,
                    "severity": check.severity.value,
                    "description": check.description
                }
                for check in standard_checks if check.status == "fail"
            ],
            "recommendations": self._generate_recommendations(standard_checks)
        }
        
        return report
    
    def _generate_recommendations(self, checks: List[ComplianceCheck]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        failed_checks = [check for check in checks if check.status == "fail"]
        
        for check in failed_checks:
            if "权限" in check.control_name:
                recommendations.append(f"修复{check.control_id}：调整文件和目录权限")
            elif "认证" in check.control_name:
                recommendations.append(f"修复{check.control_id}：更新认证配置")
            elif "监控" in check.control_name:
                recommendations.append(f"修复{check.control_id}：启用系统监控")
            else:
                recommendations.append(f"修复{check.control_id}：{check.control_name}")
        
        return recommendations[:10]  # 最多返回10条建议

def demonstrate_security_monitoring_and_compliance():
    """演示安全监控与合规性管理"""
    print("=" * 60)
    print("Kubernetes 安全监控与合规性管理演示")
    print("=" * 60)
    
    # 创建安全监控器
    monitor = SecurityMonitor()
    
    # 模拟安全事件
    print("\n--- 安全事件收集 ---")
    
    # 模拟多次认证失败 (可能为暴力破解)
    for i in range(6):
        monitor.collect_security_event(
            event_type=EventType.AUTHENTICATION_FAILURE,
            severity=Severity.MEDIUM,
            source="192.168.1.100",
            target="kube-apiserver",
            description=f"认证失败尝试 #{i+1}",
            metadata={"username": "admin", "attempts": i+1}
        )
    
    # 模拟一次权限提升尝试
    monitor.collect_security_event(
        event_type=EventType.UNAUTHORIZED_ACCESS,
        severity=Severity.HIGH,
        source="pod:malicious-app",
        target="kube-system namespace",
        description="尝试访问受保护的命名空间",
        metadata={"requested_resource": "secrets", "action": "get"}
    )
    
    # 模拟策略违规
    monitor.collect_security_event(
        event_type=EventType.POLICY_VIOLATION,
        severity=Severity.MEDIUM,
        source="pod:web-app",
        target="default namespace",
        description="容器试图写入root文件系统",
        metadata={"violation_type": "filesystem", "path": "/"}
    )
    
    # 威胁趋势分析
    print("\n--- 威胁趋势分析 ---")
    trends = monitor.analyze_threat_trends()
    print(f"总事件数: {trends['total_events']}")
    print(f"事件类型分布: {trends['event_type_distribution']}")
    print(f"严重级别分布: {trends['severity_distribution']}")
    
    # 创建合规性管理器
    compliance_mgr = ComplianceManager()
    
    # 运行CIS基准检查
    print("\n--- CIS基准合规性检查 ---")
    cis_checks = compliance_mgr.run_compliance_check(ComplianceStandard.CIS_BENCHMARK)
    
    for check in cis_checks:
        status_icon = "✅" if check.status == "pass" else "❌" if check.status == "fail" else "⚠️"
        print(f"{status_icon} {check.control_id}: {check.control_name} [{check.status}]")
    
    # 运行NIST框架检查
    print("\n--- NIST框架合规性检查 ---")
    nist_checks = compliance_mgr.run_compliance_check(ComplianceStandard.NIST_FRAMEWORK)
    
    for check in nist_checks:
        status_icon = "✅" if check.status == "pass" else "❌" if check.status == "fail" else "⚠️"
        print(f"{status_icon} {check.control_id}: {check.control_name} [{check.status}]")
    
    # 生成合规性报告
    print("\n--- 合规性报告生成 ---")
    
    # CIS基准报告
    cis_report = compliance_mgr.generate_compliance_report(ComplianceStandard.CIS_BENCHMARK)
    print(f"\nCIS基准合规性报告:")
    print(f"合规性分数: {cis_report['compliance_score']}%")
    print(f"检查项目: {cis_report['total_checks']}")
    print(f"通过项目: {cis_report['passed_checks']}")
    print(f"失败项目: {cis_report['failed_checks']}")
    print(f"警告项目: {cis_report['warning_checks']}")
    
    if cis_report['failed_controls']:
        print("失败的合规性控制:")
        for control in cis_report['failed_controls'][:3]:  # 显示前3个
            print(f"  • {control['control_id']}: {control['control_name']}")
    
    # NIST框架报告
    nist_report = compliance_mgr.generate_compliance_report(ComplianceStandard.NIST_FRAMEWORK)
    print(f"\nNIST框架合规性报告:")
    print(f"合规性分数: {nist_report['compliance_score']}%")
    print(f"检查项目: {nist_report['total_checks']}")
    print(f"通过项目: {nist_report['passed_checks']}")
    print(f"失败项目: {nist_report['failed_checks']}")
    
    # 改进建议
    print("\n--- 改进建议 ---")
    all_recommendations = cis_report['recommendations'] + nist_report['recommendations']
    for i, recommendation in enumerate(all_recommendations[:5], 1):  # 显示前5条
        print(f"  {i}. {recommendation}")
    
    print("\n安全监控与合规性管理最佳实践:")
    best_practices = [
        "实施实时安全事件监控",
        "配置安全事件告警规则",
        "定期进行威胁猎杀",
        "建立安全事件响应流程",
        "运行自动化合规性检查",
        "定期审查安全策略",
        "集成威胁情报",
        "建立基线行为模型",
        "实施零信任架构",
        "定期进行渗透测试"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")

# 运行演示
demonstrate_security_monitoring_and_compliance()
```) 
                                                   for key in self.keys.values()]
                        ]
                    }
                }
            ]
        else:
            providers = [
                {
                    "identity": {}
                },
                {
                    provider.value: {
                        "keys": [
                            {
                                "name": key_id,
                                "secret": key_data
                            }
                            for key_id, key_data in [(key.key_id, key.key_data