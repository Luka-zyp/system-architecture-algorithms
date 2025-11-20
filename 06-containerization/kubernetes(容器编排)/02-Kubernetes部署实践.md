# Kubernetes部署实践与配置管理

## 目录
1. [部署策略](#部署策略)
2. [配置管理](#配置管理)
3. [存储管理](#存储管理)
4. [网络管理](#网络管理)
5. [资源管理](#资源管理)
6. [安全策略](#安全策略)
7. [监控与日志](#监控与日志)
8. [故障排除](#故障排除)

## 部署策略

### 滚动更新与蓝绿部署
```python
from typing import List, Dict, Any
import time
import yaml
import json

class DeploymentStrategies:
    """Kubernetes部署策略实现"""
    
    def __init__(self):
        self.strategies = {
            'rolling_update': self._rolling_update_strategy,
            'blue_green': self._blue_green_strategy,
            'canary': self._canary_strategy,
            'recreate': self._recreate_strategy
        }
    
    def _rolling_update_strategy(self):
        """滚动更新策略"""
        return {
            'description': '渐进式替换旧版本Pod为新版本',
            'advantages': [
                '零停机时间',
                '支持自动回滚',
                '资源效率高',
                '风险可控'
            ],
            'configuration': {
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxSurge': '25%',        # 最多超出的Pod数量
                        'maxUnavailable': '25%'   # 最多不可用的Pod数量
                    }
                },
                'readiness_probe': '健康检查',
                'liveness_probe': '存活检查'
            },
            'implementation': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
        version: v2
    spec:
      containers:
      - name: web-app
        image: web-app:v2.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
            ''',
            'monitoring': {
                'metrics': [
                    'Pod状态变化',
                    '滚动更新进度',
                    '健康检查成功率',
                    '请求错误率'
                ],
                'alerts': [
                    '更新超时',
                    '健康检查失败',
                    '错误率过高'
                ]
            }
        }
    
    def _blue_green_strategy(self):
        """蓝绿部署策略"""
        return {
            'description': '维护两个完全相同的环境，切换流量',
            'advantages': [
                '快速回滚',
                '完整测试新版本',
                '零停机时间',
                '易于故障隔离'
            ],
            'disadvantages': [
                '资源成本高',
                '需要双倍基础设施',
                '数据库迁移复杂'
            ],
            'implementation_steps': [
                '1. 部署绿色环境',
                '2. 验证绿色环境健康',
                '3. 切换服务到绿色环境',
                '4. 监控运行状态',
                '5. 清理蓝色环境'
            ],
            'yaml_manifests': {
                'blue_deployment': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app-blue
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: web-app
      version: blue
  template:
    metadata:
      labels:
        app: web-app
        version: blue
    spec:
      containers:
      - name: web-app
        image: web-app:v1.0
        ports:
        - containerPort: 8080
                ''',
                'green_deployment': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app-green
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: web-app
      version: green
  template:
    metadata:
      labels:
        app: web-app
        version: green
    spec:
      containers:
      - name: web-app
        image: web-app:v2.0
        ports:
        - containerPort: 8080
                ''',
                'service': '''
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
  namespace: production
spec:
  selector:
    app: web-app
    version: blue    # 切换时修改为 green
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
                '''
            },
            'traffic_switching': {
                'service_selector': '切换Service的选择器标签',
                'istio_virtualservice': '使用Istio进行流量路由',
                'ingress_controller': '通过Ingress进行流量切换'
            }
        }
    
    def _canary_strategy(self):
        """金丝雀部署策略"""
        return {
            'description': '将少量流量路由到新版本，逐步扩大',
            'advantages': [
                '降低风险',
                '真实环境测试',
                '可观察性强',
                '成本可控'
            ],
            'phases': [
                'Phase 1: 5% 流量到新版本',
                'Phase 2: 25% 流量到新版本',
                'Phase 3: 50% 流量到新版本',
                'Phase 4: 100% 流量到新版本'
            ],
            'implementation': {
                'istio_canary': '''
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: web-app
spec:
  http:
  - route:
    - destination:
        host: web-app
        subset: v1
      weight: 95
    - destination:
        host: web-app
        subset: v2
      weight: 5
            ''',
                'nginx_canary': '''
http {
    upstream web_app_v1 {
        server web-app-v1:8080;
    }
    
    upstream web_app_v2 {
        server web-app-v2:8080;
    }
    
    server {
        location / {
            if ($remote_addr ~* "^(.*\.)?example\.com$") {
                set $canary_version "v2";
            }
            
            if ($canary_version = "v2") {
                proxy_pass http://web_app_v2;
            } else {
                proxy_pass http://web_app_v1;
            }
        }
    }
}
            ''',
                'kubernetes_hpa': '''
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-canary
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app-v2
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
            '''
            },
            'monitoring_metrics': [
                '错误率变化',
                '响应时间变化',
                '吞吐量变化',
                '用户体验指标'
            ],
            'rollback_triggers': [
                '错误率超过阈值',
                '响应时间急剧增加',
                '新版本健康检查失败',
                '用户投诉增加'
            ]
        }
    
    def _recreate_strategy(self):
        """重建策略"""
        return {
            'description': '先删除旧版本Pod，再创建新版本',
            'advantages': [
                '实现简单',
                '配置要求低',
                '适用于数据迁移场景'
            ],
            'disadvantages': [
                '有停机时间',
                '不适合高可用要求',
                '用户访问中断'
            ],
            'use_cases': [
                '开发测试环境',
                '离线维护窗口',
                '数据库迁移',
                '紧急修复'
            ],
            'implementation': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  strategy:
    type: Recreate
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: web-app:v2.0
        ports:
        - containerPort: 8080
            '''
        }

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self):
        self.strategies = DeploymentStrategies()
        self.deployment_history = []
    
    def plan_deployment(self, current_version: str, target_version: str, 
                       traffic_volume: int, risk_tolerance: str) -> Dict[str, Any]:
        """规划部署策略"""
        strategy_selection = {
            'low_risk': {
                'strategy': 'rolling_update',
                'canary_percentage': 10,
                'monitoring_duration': 300,  # 5分钟
                'rollback_threshold': 0.05   # 5%错误率
            },
            'medium_risk': {
                'strategy': 'canary',
                'canary_percentage': 25,
                'phases': [5, 25, 50, 100],
                'monitoring_duration': 1800,  # 30分钟
                'rollback_threshold': 0.03   # 3%错误率
            },
            'high_risk': {
                'strategy': 'blue_green',
                'validation_time': 3600,      # 1小时
                'traffic_switch_method': 'service_selector'
            }
        }
        
        if risk_tolerance in strategy_selection:
            return {
                'current_version': current_version,
                'target_version': target_version,
                'traffic_volume': traffic_volume,
                'recommended_strategy': strategy_selection[risk_tolerance],
                'timeline': self._estimate_timeline(risk_tolerance),
                'resources_needed': self._estimate_resources(risk_tolerance),
                'risk_assessment': self._assess_risk(current_version, target_version)
            }
        else:
            raise ValueError(f"Unknown risk tolerance: {risk_tolerance}")
    
    def _estimate_timeline(self, risk_tolerance: str) -> Dict[str, int]:
        """估算部署时间线"""
        timelines = {
            'low_risk': {
                'preparation': 15,    # 15分钟准备
                'deployment': 10,     # 10分钟部署
                'monitoring': 15,     # 15分钟监控
                'total': 40
            },
            'medium_risk': {
                'preparation': 30,
                'deployment': 45,
                'monitoring': 60,
                'total': 135
            },
            'high_risk': {
                'preparation': 60,
                'deployment': 30,
                'validation': 60,
                'switch_traffic': 15,
                'total': 165
            }
        }
        return timelines[risk_tolerance]
    
    def _estimate_resources(self, risk_tolerance: str) -> Dict[str, int]:
        """估算资源需求"""
        resource_requirements = {
            'low_risk': {
                'additional_pods': 1,
                'additional_memory': '128Mi per pod',
                'additional_cpu': '100m per pod'
            },
            'medium_risk': {
                'additional_pods': 2,
                'additional_memory': '256Mi per pod',
                'additional_cpu': '200m per pod'
            },
            'high_risk': {
                'additional_pods': 4,  # 蓝绿环境需要双倍资源
                'additional_memory': '512Mi per pod',
                'additional_cpu': '500m per pod'
            }
        }
        return resource_requirements[risk_tolerance]
    
    def _assess_risk(self, current_version: str, target_version: str) -> Dict[str, str]:
        """风险评估"""
        risk_factors = {
            'schema_changes': 'database schema compatibility',
            'api_changes': 'backward compatibility',
            'infrastructure_changes': 'dependency version changes',
            'configuration_changes': 'environment configuration changes'
        }
        
        risk_level = 'medium'  # 默认中等风险
        
        if self._is_major_version_change(current_version, target_version):
            risk_level = 'high'
        elif self._is_minor_version_change(current_version, target_version):
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'factors': risk_factors,
            'mitigation_strategies': [
                '使用金丝雀部署',
                '增加监控频率',
                '准备快速回滚计划',
                '通知利益相关者'
            ]
        }
    
    def _is_major_version_change(self, current: str, target: str) -> bool:
        """检查是否为主版本变更"""
        try:
            current_major = int(current.split('.')[0].replace('v', ''))
            target_major = int(target.split('.')[0].replace('v', ''))
            return target_major > current_major
        except:
            return False
    
    def _is_minor_version_change(self, current: str, target: str) -> bool:
        """检查是否为次版本变更"""
        try:
            current_parts = current.split('.')
            target_parts = target.split('.')
            
            current_major = int(current_parts[0].replace('v', ''))
            target_major = int(target_parts[0].replace('v', ''))
            
            if current_major == target_major:
                return len(current_parts) > 1 and len(target_parts) > 1 and \
                       int(target_parts[1]) > int(current_parts[1])
            return False
        except:
            return False

# 使用示例
async def demo_deployment_strategies():
    """演示部署策略"""
    print("\n=== Kubernetes部署策略演示 ===")
    
    # 部署策略分析
    strategies = DeploymentStrategies()
    
    for strategy_name, strategy_func in strategies.strategies.items():
        strategy_info = strategy_func()
        print(f"\n{strategy_name.upper()}:")
        print(f"  描述: {strategy_info['description']}")
        
        if 'advantages' in strategy_info:
            print("  优点:")
            for advantage in strategy_info['advantages']:
                print(f"    - {advantage}")
        
        if 'disadvantages' in strategy_info:
            print("  缺点:")
            for disadvantage in strategy_info['disadvantages']:
                print(f"    - {disadvantage}")
    
    # 部署规划演示
    manager = DeploymentManager()
    deployment_plan = manager.plan_deployment(
        current_version="v1.2.3",
        target_version="v1.3.0", 
        traffic_volume=10000,
        risk_tolerance="medium"
    )
    
    print(f"\n\n=== 部署计划 ===")
    print(f"当前版本: {deployment_plan['current_version']}")
    print(f"目标版本: {deployment_plan['target_version']}")
    print(f"推荐策略: {deployment_plan['recommended_strategy']['strategy']}")
    print(f"风险级别: {deployment_plan['risk_assessment']['risk_level']}")
    
    timeline = deployment_plan['timeline']
    print(f"\n部署时间线:")
    for phase, duration in timeline.items():
        print(f"  {phase}: {duration}分钟")
    
    resources = deployment_plan['resources_needed']
    print(f"\n资源需求:")
    for resource, amount in resources.items():
        print(f"  {resource}: {amount}")

# 运行演示
import asyncio
asyncio.run(demo_deployment_strategies())
```

## 配置管理

### ConfigMap与Secret管理
```python
from typing import Dict, Any, List
import base64
import json
import yaml

class ConfigManagement:
    """配置管理工具"""
    
    def __init__(self):
        self.config_types = {
            'configmap': self._manage_configmap,
            'secret': self._manage_secret,
            'environment_variables': self._manage_env_vars,
            'volume_mounts': self._manage_volume_mounts
        }
    
    def _manage_configmap(self):
        """ConfigMap管理"""
        return {
            'purpose': '存储非敏感配置数据',
            'creation_methods': {
                'from_literals': {
                    'description': '从字面量创建',
                    'example': '''
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_host: "localhost"
  database_port: "5432"
  log_level: "INFO"
                    '''
                },
                'from_file': {
                    'description': '从文件创建',
                    'commands': [
                        'kubectl create configmap app-config \\',
                        '  --from-file=app.properties',
                        'kubectl create configmap nginx-config \\',
                        '  --from-file=nginx.conf'
                    ]
                },
                'from_env_file': {
                    'description': '从环境变量文件创建',
                    'commands': [
                        'kubectl create configmap app-env \\',
                        '  --from-env-file=.env'
                    ]
                },
                'from_directory': {
                    'description': '从目录创建',
                    'commands': [
                        'kubectl create configmap configs \\',
                        '  --from-file=config/'
                    ]
                }
            },
            'usage_patterns': {
                'environment_variables': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    env:
    - name: DATABASE_HOST
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_host
  restartPolicy: Always
                ''',
                'volume_mounts': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
  volumes:
  - name: config-volume
    configMap:
      name: app-config
  restartPolicy: Always
                ''',
                'command_arguments': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    args:
    - --config=$(CONFIG_FILE)
    env:
    - name: CONFIG_FILE
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: config_file
  restartPolicy: Always
                '''
            },
            'dynamic_updates': {
                'watch_k8s_api': '监听ConfigMap变化',
                'hot_reload': '应用热重载配置',
                'graceful_restart': '优雅重启应用',
                'health_checks': '验证配置更新'
            }
        }
    
    def _manage_secret(self):
        """Secret管理"""
        return {
            'purpose': '存储敏感数据（密码、令牌、密钥）',
            'secret_types': {
                'opaque': {
                    'description': '不透明的秘密',
                    'encoding': 'base64编码',
                    'example': '''
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  password: cGFzc3dvcmQxMjM=    # password123
  api_key: YWJjZGVmZ2hpams=   # abcdefghijk
                    '''
                },
                'kubernetes.io/dockerconfigjson': {
                    'description': 'Docker镜像仓库认证',
                    'creation_command': 'kubectl create secret docker-registry my-registry \\',
                    '                --docker-username=username \\',
                    '                --docker-password=password \\',
                    '                --docker-email=email@example.com'
                },
                'kubernetes.io/tls': {
                    'description': 'TLS证书',
                    'creation_command': 'kubectl create secret tls my-tls \\',
                    '                --cert=path/to/cert.pem \\',
                    '                --key=path/to/key.pem'
                }
            },
            'security_best_practices': [
                '使用TLS证书',
                '定期轮换密钥',
                '最小权限原则',
                '审计Secret访问',
                '避免在镜像中硬编码密钥'
            ],
            'usage_patterns': {
                'environment_variables': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    env:
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: app-secrets
          key: password
  restartPolicy: Always
                ''',
                'volume_mounts': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true
  volumes:
  - name: secret-volume
    secret:
      secretName: app-secrets
  restartPolicy: Always
                '''
            },
            'external_secret_management': {
                'vault_integration': 'HashiCorp Vault集成',
                'aws_secrets_manager': 'AWS Secrets Manager',
                'azure_key_vault': 'Azure Key Vault',
                'gcp_secret_manager': 'Google Cloud Secret Manager'
            }
        }
    
    def _manage_env_vars(self):
        """环境变量管理"""
        return {
            'static_values': {
                'description': '静态环境变量',
                'example': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    env:
    - name: ENVIRONMENT
      value: "production"
    - name: LOG_LEVEL
      value: "INFO"
    - name: TIMEOUT
      value: "30"
                '''
            },
            'from_configmap': {
                'description': '从ConfigMap获取环境变量',
                'example': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_url
    - name: API_ENDPOINT
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: api_endpoint
                '''
            },
            'from_secret': {
                'description': '从Secret获取环境变量',
                'example': '''
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: app:latest
    env:
    - name: DATABASE_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password
    - name: API_KEY
      valueFrom:
        secretKeyRef:
          name: api-secret
          key: api_key
                '''
            },
            'dynamic_values': {
                'description': '动态计算的环境变量',
                'capabilities': [
                    '运行容器时动态生成',
                    '从其他环境变量组合',
                    '基于当前时间生成',
                    '随机值生成'
                ]
            },
            'environment_specific': {
                'development': '开发环境配置',
                'staging': '测试环境配置',
                'production': '生产环境配置',
                'disaster_recovery': '灾难恢复环境配置'
            }
        }
    
    def _manage_volume_mounts(self):
        """卷挂载管理"""
        return {
            'configmap_as_volume': {
                'description': '将ConfigMap挂载为卷',
                'example': '''
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    events {
        worker_connections 1024;
    }
    http {
        server {
            listen 80;
            location / {
                return 200 "Hello from Kubernetes";
            }
        }
    }
---
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx
    image: nginx:latest
    volumeMounts:
    - name: config-volume
      mountPath: /etc/nginx/nginx.conf
      subPath: nginx.conf
  volumes:
  - name: config-volume
    configMap:
      name: nginx-config
                '''
            },
            'secret_as_volume': {
                'description': '将Secret挂载为卷',
                'example': '''
apiVersion: v1
kind: Secret
metadata:
  name: tls-certificates
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi...
  tls.key: LS0tLS1CRUdJTi...
---
apiVersion: v1
kind: Pod
metadata:
  name: https-app
spec:
  containers:
  - name: app
    image: app:latest
    volumeMounts:
    - name: tls-volume
      mountPath: /etc/tls
      readOnly: true
  volumes:
  - name: tls-volume
    secret:
      secretName: tls-certificates
                '''
            },
            'subpath_usage': {
                'description': '使用subPath选择特定文件',
                'benefits': [
                    '避免覆盖整个目录',
                    '选择性挂载文件',
                    '减少配置复杂性'
                ],
                'example': '''
apiVersion: v1
kind: Pod
metadata:
  name: multi-config-pod
spec:
  containers:
  - name: app
    image: app:latest
    volumeMounts:
    - name: config-volume
      mountPath: /etc/app/database.conf
      subPath: database.conf
    - name: config-volume
      mountPath: /etc/app/cache.conf
      subPath: cache.conf
  volumes:
  - name: config-volume
    configMap:
      name: multi-config
                '''
            },
            'file_permissions': {
                'description': '设置文件和目录权限',
                'example': '''
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
spec:
  containers:
  - name: app
    image: app:latest
    volumeMounts:
    - name: secure-volume
      mountPath: /secure
  volumes:
  - name: secure-volume
    secret:
      secretName: secure-config
      defaultMode: 0400  # 只读权限
                '''
            }
        }

class ConfigurationValidation:
    """配置验证工具"""
    
    def __init__(self):
        self.validators = {
            'configmap_validator': self._validate_configmap,
            'secret_validator': self._validate_secret,
            'env_var_validator': self._validate_environment_vars,
            'volume_validator': self._validate_volume_mounts
        }
    
    def _validate_configmap(self, configmap_data: Dict[str, str]) -> Dict[str, Any]:
        """验证ConfigMap配置"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        for key, value in configmap_data.items():
            # 检查键名格式
            if not self._is_valid_key_name(key):
                validation_results['errors'].append(f"Invalid key name: {key}")
                validation_results['valid'] = False
            
            # 检查值长度
            if len(value) > 1024 * 1024:  # 1MB限制
                validation_results['warnings'].append(f"Large value for key: {key}")
            
            # 检查敏感数据
            if self._contains_sensitive_data(value):
                validation_results['recommendations'].append(
                    f"Consider using Secret for key: {key}"
                )
        
        return validation_results
    
    def _validate_secret(self, secret_data: Dict[str, str]) -> Dict[str, Any]:
        """验证Secret配置"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        for key, value in secret_data.items():
            # 检查值是否base64编码
            if not self._is_base64_encoded(value):
                validation_results['warnings'].append(
                    f"Value for key {key} should be base64 encoded"
                )
            
            # 检查敏感数据模式
            if self._matches_sensitive_pattern(key, value):
                validation_results['recommendations'].append(
                    f"Consider stronger encryption for: {key}"
                )
        
        return validation_results
    
    def _validate_environment_vars(self, env_vars: List[Dict[str, str]]) -> Dict[str, Any]:
        """验证环境变量配置"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        env_names = set()
        
        for env_var in env_vars:
            name = env_var.get('name', '')
            value_from = env_var.get('valueFrom', {})
            
            # 检查重复的环境变量名
            if name in env_names:
                validation_results['errors'].append(f"Duplicate environment variable: {name}")
            else:
                env_names.add(name)
            
            # 检查环境变量名格式
            if not self._is_valid_env_var_name(name):
                validation_results['errors'].append(f"Invalid environment variable name: {name}")
            
            # 检查同时设置了value和valueFrom
            if 'value' in env_var and value_from:
                validation_results['warnings'].append(
                    f"Both value and valueFrom set for: {name}"
                )
        
        return validation_results
    
    def _validate_volume_mounts(self, volume_mounts: List[Dict[str, str]]) -> Dict[str, Any]:
        """验证卷挂载配置"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        mount_paths = set()
        
        for mount in volume_mounts:
            name = mount.get('name', '')
            mount_path = mount.get('mountPath', '')
            
            # 检查重复的挂载路径
            if mount_path in mount_paths:
                validation_results['errors'].append(f"Duplicate mount path: {mount_path}")
            else:
                mount_paths.add(mount_path)
            
            # 检查挂载路径格式
            if not self._is_valid_mount_path(mount_path):
                validation_results['errors'].append(f"Invalid mount path: {mount_path}")
            
            # 检查subPath设置
            if 'subPath' in mount and mount_path.endswith('/'):
                validation_results['warnings'].append(
                    f"subPath with directory mount path: {mount_path}"
                )
        
        return validation_results
    
    def _is_valid_key_name(self, key: str) -> bool:
        """检查键名是否有效"""
        import re
        return bool(re.match(r'^[a-zA-Z0-9._-]+$', key))
    
    def _is_base64_encoded(self, value: str) -> bool:
        """检查是否为base64编码"""
        try:
            base64.b64decode(value)
            return True
        except:
            return False
    
    def _contains_sensitive_data(self, value: str) -> bool:
        """检查是否包含敏感数据"""
        sensitive_patterns = [
            r'password\s*=',
            r'api[_-]?key\s*=',
            r'secret\s*=',
            r'token\s*=',
            r'credential\s*='
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def _matches_sensitive_pattern(self, key: str, value: str) -> bool:
        """匹配敏感数据模式"""
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        
        key_lower = key.lower()
        for sensitive_key in sensitive_keys:
            if sensitive_key in key_lower and len(value) < 16:
                return True
        return False
    
    def _is_valid_env_var_name(self, name: str) -> bool:
        """检查环境变量名是否有效"""
        import re
        return bool(re.match(r'^[A-Z_][A-Z0-9_]*$', name))
    
    def _is_valid_mount_path(self, path: str) -> bool:
        """检查挂载路径是否有效"""
        return path.startswith('/') and len(path) > 1

# 使用示例
async def demo_configuration_management():
    """演示配置管理"""
    print("\n=== Kubernetes配置管理演示 ===")
    
    config_manager = ConfigManagement()
    
    # 显示配置类型
    for config_type, config_func in config_manager.config_types.items():
        config_info = config_func()
        print(f"\n{config_type.upper()}:")
        print(f"  目的: {config_info['purpose']}")
        
        if 'creation_methods' in config_info:
            print("  创建方法:")
            for method, description in config_info['creation_methods'].items():
                print(f"    - {method}: {description['description']}")
        
        if 'usage_patterns' in config_info:
            print("  使用模式:")
            for pattern, description in config_info['usage_patterns'].items():
                print(f"    - {pattern}")
    
    # 配置验证演示
    print("\n\n=== 配置验证演示 ===")
    validator = ConfigurationValidation()
    
    # 验证ConfigMap
    sample_configmap = {
        'database_host': 'localhost',
        'database_port': '5432',
        'log_level': 'INFO',
        'api_password': 'my_secret_password'  # 应该用Secret
    }
    
    validation_result = validator._validate_configmap(sample_configmap)
    print(f"ConfigMap验证结果:")
    print(f"  有效: {validation_result['valid']}")
    
    if validation_result['errors']:
        print("  错误:")
        for error in validation_result['errors']:
            print(f"    - {error}")
    
    if validation_result['warnings']:
        print("  警告:")
        for warning in validation_result['warnings']:
            print(f"    - {warning}")
    
    if validation_result['recommendations']:
        print("  建议:")
        for recommendation in validation_result['recommendations']:
            print(f"    - {recommendation}")

# 运行演示
import asyncio
asyncio.run(demo_configuration_management())
```

这个Kubernetes部署实践与配置管理文档详细介绍了：

1. **部署策略** - 滚动更新、蓝绿部署、金丝雀部署、重建策略的完整实现
2. **配置管理** - ConfigMap和Secret管理、环境变量、卷挂载的详细配置方法
3. **部署管理器** - 智能选择部署策略和资源估算
4. **配置验证工具** - 自动验证配置合规性和安全性

通过这些内容，可以深入掌握Kubernetes的部署和配置最佳实践，确保应用程序的稳定运行和安全性。