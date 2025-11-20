# Kubernetes运维监控与故障排除

## 目录
1. [监控体系](#监控体系)
2. [日志管理](#日志管理)
3. [性能调优](#性能调优)
4. [故障排除](#故障排除)
5. [安全运维](#安全运维)
6. [备份恢复](#备份恢复)
7. [自动化运维](#自动化运维)
8. [最佳实践](#最佳实践)

## 监控体系

### Prometheus + Grafana监控方案
```python
from typing import Dict, List, Any
import json
import time
import asyncio
from datetime import datetime, timedelta

class KubernetesMonitoring:
    """Kubernetes监控解决方案"""
    
    def __init__(self):
        self.monitoring_stack = {
            'prometheus': self._setup_prometheus,
            'grafana': self._setup_grafana,
            'alertmanager': self._setup_alertmanager,
            'node_exporter': self._setup_node_exporter,
            'kube_state_metrics': self._setup_kube_state_metrics
        }
    
    def _setup_prometheus(self):
        """Prometheus配置"""
        return {
            'purpose': '时序数据库，收集和存储监控数据',
            'configuration': {
                'prometheus_config': '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "kubernetes_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
    - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)
    - target_label: __address__
      replacement: kubernetes.default.svc:443
    - source_labels: [__meta_kubernetes_node_name]
      regex: (.+)
      target_label: __metrics_path__
      replacement: /api/v1/nodes/${1}/proxy/metrics

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
    - action: labelmap
      regex: __meta_kubernetes_pod_label_(.+)
    - source_labels: [__meta_kubernetes_namespace]
      action: replace
      target_label: kubernetes_namespace
    - source_labels: [__meta_kubernetes_pod_name]
      action: replace
      target_label: kubernetes_pod_name
                ''',
                'kubernetes_rules': '''
groups:
- name: kubernetes
  rules:
  - alert: PodCrashLooping
    expr: rate(kube_pod_container_status_restarts_total[15m]) * 60 * 15 > 3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Pod {{ $labels.pod }} is crash looping"
      description: "Pod {{ $labels.pod }} has been restart {{ $value }} times in the last 15 minutes"

  - alert: PodNotReady
    expr: kube_pod_status_ready{condition="false"} == 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Pod {{ $labels.pod }} is not ready"
      description: "Pod {{ $labels.pod }} has been in not ready state for more than 5 minutes"

  - alert: NodeMemoryPressure
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Node {{ $labels.instance }} memory usage is above 85%"
      description: "Node {{ $labels.instance }} memory usage is {{ $value }}%"

  - alert: NodeCPUThrottling
    expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Node {{ $labels.instance }} CPU usage is above 80%"
      description: "Node {{ $labels.instance }} CPU usage is {{ $value }}%"

  - alert: HighErrorRate
    expr: rate(nginx_http_requests_total{status=~"5.."}[5m]) / rate(nginx_http_requests_total[5m]) * 100 > 5
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High 5xx error rate detected"
      description: "Error rate is {{ $value }}% in the last 5 minutes"
                '''
            },
            'metrics_collection': {
                'cluster_metrics': [
                    'kubernetes_pods_total',
                    'kubernetes_nodes_total',
                    'kubernetes_deployments_replicas',
                    'kubernetes_services_total'
                ],
                'node_metrics': [
                    'node_cpu_seconds_total',
                    'node_memory_MemTotal_bytes',
                    'node_memory_MemAvailable_bytes',
                    'node_disk_io_time_seconds_total',
                    'node_network_receive_bytes_total',
                    'node_network_transmit_bytes_total'
                ],
                'pod_metrics': [
                    'kube_pod_container_status_restarts_total',
                    'kube_pod_container_status_ready',
                    'kube_pod_container_resource_limits',
                    'kube_pod_container_resource_requests'
                ],
                'application_metrics': [
                    'nginx_http_requests_total',
                    'response_time_seconds',
                    'database_connections_active',
                    'cache_hit_ratio'
                ]
            },
            'query_examples': [
                {
                    'name': 'Pod CPU使用率',
                    'query': 'sum(rate(container_cpu_usage_seconds_total{namespace="default"}[5m])) by (pod) * 100',
                    'description': '查询默认命名空间中Pod的CPU使用率'
                },
                {
                    'name': '内存使用率',
                    'query': 'sum(container_memory_usage_bytes{namespace="default"}) by (pod) / sum(container_spec_memory_limit_bytes{namespace="default"}) by (pod) * 100',
                    'description': '计算Pod内存使用率百分比'
                },
                {
                    'name': '网络流量',
                    'query': 'sum(rate(container_network_receive_bytes_total[5m])) by (pod)',
                    'description': 'Pod网络接收流量'
                }
            ]
        }
    
    def _setup_grafana(self):
        """Grafana配置"""
        return {
            'purpose': '可视化监控面板，提供丰富的图表和仪表板',
            'dashboard_templates': {
                'cluster_overview': '''
{
  "dashboard": {
    "title": "Kubernetes Cluster Overview",
    "panels": [
      {
        "title": "Cluster CPU Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total[5m])) * 100",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "title": "Cluster Memory Usage",
        "type": "stat", 
        "targets": [
          {
            "expr": "sum(container_memory_usage_bytes) / sum(node_memory_MemTotal_bytes) * 100",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "Pod Status",
        "type": "table",
        "targets": [
          {
            "expr": "kube_pod_status_phase",
            "legendFormat": "{{ phase }}"
          }
        ]
      }
    ]
  }
}
                ''',
                'node_details': '''
{
  "dashboard": {
    "title": "Node Details",
    "panels": [
      {
        "title": "CPU Usage by Node",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{ instance }}"
          }
        ]
      },
      {
        "title": "Memory Usage by Node", 
        "type": "graph",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "{{ instance }}"
          }
        ]
      },
      {
        "title": "Disk Usage by Node",
        "type": "graph", 
        "targets": [
          {
            "expr": "100 - (avg(node_filesystem_avail_bytes{mountpoint!~\".*tmp.*\"}) / avg(node_filesystem_size_bytes{mountpoint!~\".*tmp.*\"}) * 100)",
            "legendFormat": "{{ instance }}"
          }
        ]
      }
    ]
  }
}
                ''',
                'pod_details': '''
{
  "dashboard": {
    "title": "Pod Details",
    "panels": [
      {
        "title": "Pod CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total[5m])) by (pod, namespace)",
            "legendFormat": "{{ pod }} ({{ namespace }})"
          }
        ]
      },
      {
        "title": "Pod Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(container_memory_usage_bytes) by (pod, namespace)",
            "legendFormat": "{{ pod }} ({{ namespace }})"
          }
        ]
      },
      {
        "title": "Pod Network Traffic",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(container_network_receive_bytes_total[5m])) by (pod, namespace)",
            "legendFormat": "RX {{ pod }} ({{ namespace }})"
          }
        ]
      }
    ]
  }
}
                '''
            },
            'datasource_configuration': {
                'prometheus': {
                    'name': 'Prometheus',
                    'type': 'prometheus',
                    'url': 'http://prometheus:9090',
                    'access': 'proxy',
                    'isDefault': True,
                    'jsonData': {
                        'httpMethod': 'POST',
                        'queryTimeout': '60s',
                        'timeInterval': '15s'
                    }
                }
            },
            'alert_integration': {
                'grafana_alerts': 'Grafana内置告警',
                'alertmanager_integration': '与AlertManager集成',
                'notification_channels': [
                    'Email',
                    'Slack',
                    'PagerDuty',
                    '钉钉/企业微信'
                ]
            }
        }
    
    def _setup_alertmanager(self):
        """AlertManager配置"""
        return {
            'purpose': '告警管理，处理和路由告警通知',
            'configuration': {
                'alertmanager_config': '''
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@example.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://example.com/webhook'
    
- name: 'critical-alerts'
  email_configs:
  - to: 'ops-team@example.com'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Severity: {{ .Labels.severity }}
      Instance: {{ .Labels.instance }}
      Time: {{ .StartsAt.Format "2006-01-02 15:04:05" }}
      {{ end }}
    
- name: 'warning-alerts'
  email_configs:
  - to: 'dev-team@example.com'
    subject: 'WARNING: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Severity: {{ .Labels.severity }}
      Instance: {{ .Labels.instance }}
      Time: {{ .StartsAt.Format "2006-01-02 15:04:05" }}
      {{ end }}
                ''',
                'silence_config': '''
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-silences
data:
  maintenance-window.yaml: |
    route:
      receiver: 'default'
      group_by: ['alertname']
    receivers:
    - name: 'default'
      webhook_configs:
      - url: 'http://maintenance-system/alerts'
                '''
            },
            'escalation_rules': [
                {
                    'level': 1,
                    'condition': 'Critical告警持续5分钟',
                    'action': '发送邮件和Slack通知'
                },
                {
                    'level': 2,
                    'condition': 'Critical告警持续15分钟',
                    'action': '发送短信和电话通知'
                },
                {
                    'level': 3,
                    'condition': 'Critical告警持续30分钟',
                    'action': '触发值班轮换和自动恢复流程'
                }
            ]
        }
    
    def _setup_node_exporter(self):
        """Node Exporter配置"""
        return {
            'purpose': '收集节点级别的系统指标',
            'daemonset_manifest': '''
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9100"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.6.1
        ports:
        - containerPort: 9100
          hostPort: 9100
        volumeMounts:
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: dev
          mountPath: /host/dev
          readOnly: true
        args:
        - --path.procfs=/host/proc
        - --path.sysfs=/host/sys
        - --collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)
      volumes:
      - name: sys
        hostPath:
          path: /sys
      - name: proc
        hostPath:
          path: /proc
      - name: dev
        hostPath:
          path: /dev
            ''',
            'collectors': {
                'system_metrics': [
                    'cpu-info',
                    'memory-info',
                    'disk-usage',
                    'network-stats',
                    'filesystem-stats'
                ],
                'custom_collectors': [
                    'nvidia-smi',
                    'custom-app-metrics',
                    'log-metrics'
                ]
            }
        }
    
    def _setup_kube_state_metrics(self):
        """Kube State Metrics配置"""
        return {
            'purpose': '收集Kubernetes集群状态指标',
            'deployment_manifest': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kube-state-metrics
  namespace: monitoring
  labels:
    app: kube-state-metrics
spec:
  selector:
    matchLabels:
      app: kube-state-metrics
  template:
    metadata:
      labels:
        app: kube-state-metrics
    spec:
      serviceAccountName: kube-state-metrics
      containers:
      - name: kube-state-metrics
        image: registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.9.2
        ports:
        - containerPort: 8080
        - containerPort: 8081
        args:
        - --host=127.0.0.1
        - --port=8080
        - --telemetry-host=127.0.0.1
        - --telemetry-port=8081
        resources:
          requests:
            cpu: 10m
            memory: 50Mi
          limits:
            cpu: 100m
            memory: 100Mi
      - name: addon-resizer
        image: k8s.gcr.io/addon-resizer:7.1.0
        resources:
          requests:
            cpu: 10m
            memory: 50Mi
          limits:
            cpu: 100m
            memory: 50Mi
        env:
        - name: MY_POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: MY_POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: MY_POD_UID
          valueFrom:
            fieldRef:
              fieldPath: metadata.uid
        - name: TARGET_MEMORY_MIB
          value: "100"
        - name: TARGET_CPU_MILLICORES
          value: "50"
        volumeMounts:
        - name: config
          mountPath: /etc/config
      volumes:
      - name: config
        configMap:
          name: kube-state-metrics-config
            ''',
            'metrics': {
                'cluster_state': [
                    'kube_deployment_status_replicas',
                    'kube_deployment_status_replicas_available',
                    'kube_statefulset_status_replicas',
                    'kube_daemonset_status_desired_number_scheduled'
                ],
                'resource_usage': [
                    'kube_pod_container_resource_limits',
                    'kube_pod_container_resource_requests',
                    'kube_node_status_allocatable',
                    'kube_node_status_capacity'
                ],
                'pod_info': [
                    'kube_pod_status_phase',
                    'kube_pod_status_ready',
                    'kube_pod_container_status_waiting',
                    'kube_pod_container_status_restarts_total'
                ]
            }
        }

class MonitoringMetricsAnalyzer:
    """监控指标分析器"""
    
    def __init__(self):
        self.analyzers = {
            'resource_analyzer': self._analyze_resource_usage,
            'performance_analyzer': self._analyze_performance,
            'anomaly_detector': self._detect_anomalies,
            'capacity_planner': self._plan_capacity
        }
    
    def _analyze_resource_usage(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析资源使用情况"""
        analysis = {
            'cpu_analysis': {
                'cluster_cpu_usage': 'sum(rate(container_cpu_usage_seconds_total[5m])) * 100',
                'node_cpu_distribution': self._calculate_cpu_distribution(metrics_data.get('node_cpu', {})),
                'pod_cpu_hotspots': self._identify_cpu_hotspots(metrics_data.get('pod_cpu', {})),
                'recommendations': [
                    'CPU使用率超过80%的节点考虑扩容',
                    'CPU使用率低于20%的节点可以缩容',
                    '识别CPU密集型Pod并优化'
                ]
            },
            'memory_analysis': {
                'cluster_memory_usage': 'sum(container_memory_usage_bytes) / sum(node_memory_MemTotal_bytes) * 100',
                'memory_leak_detection': self._detect_memory_leaks(metrics_data.get('memory_timeseries', {})),
                'oom_risk_assessment': self._assess_oom_risk(metrics_data.get('pod_memory', {})),
                'recommendations': [
                    '内存使用率超过85%的节点需要关注',
                    '检查内存使用趋势，识别潜在泄露',
                    '为高内存Pod增加limits'
                ]
            },
            'storage_analysis': {
                'disk_usage_by_node': 'node_filesystem_avail_bytes / node_filesystem_size_bytes',
                'pvc_usage_trends': self._analyze_pvc_usage(metrics_data.get('pvc_metrics', {})),
                'recommendations': [
                    '磁盘使用率超过90%的节点需要清理',
                    '监控PVC使用趋势，预留足够空间',
                    '考虑使用存储优化方案'
                ]
            }
        }
        return analysis
    
    def _analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能指标"""
        return {
            'response_time_analysis': {
                'p50_latency': 'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
                'p95_latency': 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
                'p99_latency': 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
                'trends': self._analyze_latency_trends(performance_data.get('latency_data', []))
            },
            'throughput_analysis': {
                'requests_per_second': 'sum(rate(http_requests_total[5m]))',
                'success_rate': 'sum(rate(http_requests_total{status!~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100',
                'hot_endpoints': self._identify_hot_endpoints(performance_data.get('endpoint_metrics', {}))
            },
            'recommendations': [
                'P99响应时间超过1秒需要性能优化',
                '错误率超过1%需要立即调查',
                '识别瓶颈节点并优化负载分布'
            ]
        }
    
    def _detect_anomalies(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """异常检测"""
        anomaly_detection = {
            'statistical_anomalies': [
                {
                    'type': 'sudden_spike',
                    'description': '指标值突然激增',
                    'detection_method': 'z-score > 3',
                    'examples': ['CPU使用率突然增加', '网络流量异常增长']
                },
                {
                    'type': 'sudden_drop',
                    'description': '指标值突然下降',
                    'detection_method': 'z-score < -3',
                    'examples': ['Pod数量突然减少', '吞吐量急剧下降']
                },
                {
                    'type': 'trend_change',
                    'description': '趋势发生改变',
                    'detection_method': 'linear regression slope change',
                    'examples': ['内存使用呈上升趋势', '错误率持续增加']
                }
            ],
            'pattern_anomalies': [
                {
                    'type': 'seasonal_violation',
                    'description': '违反了季节性模式',
                    'detection_method': '季节性模型残差分析',
                    'examples': ['工作时间流量模式异常', '周末CPU使用模式变化']
                }
            ],
            'threshold_anomalies': [
                {
                    'type': 'hard_threshold',
                    'description': '超过硬阈值',
                    'detection_method': 'fixed threshold comparison',
                    'examples': ['磁盘使用率超过95%', '内存使用率超过90%']
                },
                {
                    'type': 'soft_threshold',
                    'description': '接近软阈值',
                    'detection_method': 'threshold * 0.9',
                    'examples': ['CPU使用率接近90%', 'Pod重启次数接近告警阈值']
                }
            ]
        }
        
        return {
            'detected_anomalies': self._scan_for_anomalies(time_series_data),
            'confidence_score': self._calculate_anomaly_confidence(time_series_data),
            'recommended_actions': self._suggest_anomaly_actions(time_series_data)
        }
    
    def _plan_capacity(self, current_usage: Dict[str, Any], growth_rate: float, time_horizon: int) -> Dict[str, Any]:
        """容量规划"""
        return {
            'cpu_capacity_planning': {
                'current_utilization': current_usage.get('cpu_utilization', 0),
                'projected_usage': current_usage.get('cpu_utilization', 0) * (1 + growth_rate) ** time_horizon,
                'recommended_additional_cpu': max(0, current_usage.get('cpu_utilization', 0) * (1 + growth_rate) ** time_horizon - 80),
                'cost_estimate': 'additional CPU cores * cost per core * time period'
            },
            'memory_capacity_planning': {
                'current_utilization': current_usage.get('memory_utilization', 0),
                'projected_usage': current_usage.get('memory_utilization', 0) * (1 + growth_rate) ** time_horizon,
                'recommended_additional_memory': max(0, current_usage.get('memory_utilization', 0) * (1 + growth_rate) ** time_horizon - 85),
                'cost_estimate': 'additional memory * cost per GB * time period'
            },
            'storage_capacity_planning': {
                'current_utilization': current_usage.get('storage_utilization', 0),
                'projected_usage': current_usage.get('storage_utilization', 0) * (1 + growth_rate) ** time_horizon,
                'recommended_additional_storage': max(0, current_usage.get('storage_utilization', 0) * (1 + growth_rate) ** time_horizon - 90),
                'cost_estimate': 'additional storage * cost per GB * time period'
            },
            'recommendations': [
                f'基于{growth_rate*100:.1f}%增长率的{time_horizon}个月容量规划',
                '考虑峰值时段的额外余量',
                '制定分阶段的扩容计划',
                '监控实际增长情况并调整预测'
            ]
        }
    
    def _calculate_cpu_distribution(self, node_cpu_data: Dict[str, Any]) -> Dict[str, float]:
        """计算CPU分布"""
        distribution = {}
        total_cpu = sum(node_cpu_data.values())
        
        for node, usage in node_cpu_data.items():
            distribution[node] = (usage / total_cpu) * 100 if total_cpu > 0 else 0
        
        return distribution
    
    def _identify_cpu_hotspots(self, pod_cpu_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别CPU热点"""
        hotspots = []
        threshold = 80.0  # 80% CPU使用率阈值
        
        for pod, usage in pod_cpu_data.items():
            if usage > threshold:
                hotspots.append({
                    'pod': pod,
                    'cpu_usage': usage,
                    'severity': 'high' if usage > 90 else 'medium',
                    'recommendation': '考虑优化或增加资源限制'
                })
        
        return hotspots
    
    def _detect_memory_leaks(self, memory_timeseries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测内存泄露"""
        leaks = []
        # 简化的内存泄露检测逻辑
        for pod, timeseries in memory_timeseries.items():
            if len(timeseries) > 10:  # 至少需要10个数据点
                # 计算趋势
                values = [point['value'] for point in timeseries[-10:]]
                if self._is_increasing_trend(values):
                    leaks.append({
                        'pod': pod,
                        'trend': 'increasing',
                        'growth_rate': self._calculate_growth_rate(values),
                        'severity': 'high' if self._calculate_growth_rate(values) > 0.1 else 'medium'
                    })
        
        return leaks
    
    def _assess_oom_risk(self, pod_memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估OOM风险"""
        oom_risks = []
        risk_threshold = 0.9  # 90%内存使用率
        
        for pod, memory_data in pod_memory.items():
            usage_ratio = memory_data.get('usage', 0) / memory_data.get('limit', 1)
            if usage_ratio > risk_threshold:
                oom_risks.append({
                    'pod': pod,
                    'memory_usage_ratio': usage_ratio,
                    'risk_level': 'high' if usage_ratio > 0.95 else 'medium',
                    'recommendation': '考虑增加内存限制或优化内存使用'
                })
        
        return oom_risks
    
    def _analyze_pvc_usage(self, pvc_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析PVC使用情况"""
        analysis = {
            'high_usage_pvcs': [],
            'low_usage_pvcs': [],
            'growth_trends': {}
        }
        
        for pvc, metrics in pvc_metrics.items():
            usage_ratio = metrics.get('used', 0) / metrics.get('capacity', 1)
            
            if usage_ratio > 0.8:
                analysis['high_usage_pvcs'].append({
                    'pvc': pvc,
                    'usage_ratio': usage_ratio,
                    'recommendation': '考虑扩容或清理数据'
                })
            elif usage_ratio < 0.3:
                analysis['low_usage_pvcs'].append({
                    'pvc': pvc,
                    'usage_ratio': usage_ratio,
                    'recommendation': '考虑缩容以节省成本'
                })
        
        return analysis
    
    def _analyze_latency_trends(self, latency_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析延迟趋势"""
        if not latency_data:
            return {'trend': 'no_data', 'analysis': 'insufficient_data'}
        
        recent_latencies = [d['latency'] for d in latency_data[-10:]]  # 最近10个数据点
        
        if len(recent_latencies) >= 2:
            # 简单线性趋势分析
            recent_avg = sum(recent_latencies[-5:]) / min(5, len(recent_latencies))
            previous_avg = sum(recent_latencies[:5]) / min(5, len(recent_latencies))
            
            if recent_avg > previous_avg * 1.2:
                trend = 'increasing'
                analysis = 'latency_is_increasing'
            elif recent_avg < previous_avg * 0.8:
                trend = 'decreasing'
                analysis = 'latency_is_decreasing'
            else:
                trend = 'stable'
                analysis = 'latency_is_stable'
        else:
            trend = 'insufficient_data'
            analysis = 'need_more_data_points'
        
        return {'trend': trend, 'analysis': analysis}
    
    def _identify_hot_endpoints(self, endpoint_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别热点端点"""
        hot_endpoints = []
        sorted_endpoints = sorted(endpoint_metrics.items(), key=lambda x: x[1].get('requests_per_second', 0), reverse=True)
        
        for endpoint, metrics in sorted_endpoints[:5]:  # 前5个热点端点
            hot_endpoints.append({
                'endpoint': endpoint,
                'requests_per_second': metrics.get('requests_per_second', 0),
                'avg_latency': metrics.get('avg_latency', 0),
                'error_rate': metrics.get('error_rate', 0),
                'recommendation': self._suggest_endpoint_optimization(metrics)
            })
        
        return hot_endpoints
    
    def _suggest_endpoint_optimization(self, metrics: Dict[str, Any]) -> str:
        """建议端点优化"""
        rps = metrics.get('requests_per_second', 0)
        latency = metrics.get('avg_latency', 0)
        error_rate = metrics.get('error_rate', 0)
        
        if error_rate > 5:
            return '高错误率，需要调试和修复'
        elif latency > 1:
            return '延迟过高，需要性能优化'
        elif rps > 1000:
            return '流量较大，考虑缓存或负载均衡'
        else:
            return '性能正常'
    
    def _scan_for_anomalies(self, time_series_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扫描异常"""
        anomalies = []
        
        # 这里应该实现具体的异常检测算法
        # 简化示例：基于3-sigma规则
        for series in time_series_data:
            if len(series.get('values', [])) > 10:
                values = [v['value'] for v in series['values'][-20:]]  # 最近20个点
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5
                
                for i, value in enumerate(values[-5:]):  # 检查最近5个点
                    if abs(value - mean) > 3 * std_dev:
                        anomalies.append({
                            'metric': series.get('name', 'unknown'),
                            'value': value,
                            'expected_range': f'{mean - 3*std_dev} - {mean + 3*std_dev}',
                            'timestamp': values[-5:][i] if 'timestamp' in series['values'][-5:][i] else None,
                            'severity': 'high' if abs(value - mean) > 4 * std_dev else 'medium'
                        })
        
        return anomalies
    
    def _calculate_anomaly_confidence(self, time_series_data: List[Dict[str, Any]]) -> float:
        """计算异常置信度"""
        # 简化的置信度计算
        if not time_series_data:
            return 0.0
        
        total_anomalies = sum(1 for series in time_series_data if self._has_anomaly(series))
        confidence = min(1.0, total_anomalies / len(time_series_data) * 2)
        
        return confidence
    
    def _suggest_anomaly_actions(self, time_series_data: List[Dict[str, Any]]) -> List[str]:
        """建议异常处理措施"""
        actions = []
        
        # 基于异常类型给出建议
        anomaly_count = len([series for series in time_series_data if self._has_anomaly(series)])
        
        if anomaly_count == 0:
            actions.append('未检测到异常，系统运行正常')
        elif anomaly_count <= 2:
            actions.append('检测到少量异常，建议密切监控')
            actions.append('检查相关Pod和服务的状态')
        elif anomaly_count <= 5:
            actions.append('检测到多个异常，可能存在系统性问题')
            actions.append('建议立即介入调查')
            actions.append('考虑执行回滚或重启操作')
        else:
            actions.append('检测到大量异常，系统可能存在严重问题')
            actions.append('建议立即执行应急预案')
            actions.append('考虑故障转移或紧急扩容')
        
        return actions
    
    def _has_anomaly(self, series: Dict[str, Any]) -> bool:
        """检查是否有异常"""
        # 简化的异常检查逻辑
        values = [v['value'] for v in series.get('values', [])[-10:]]
        if len(values) < 5:
            return False
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        for value in values[-3:]:  # 检查最近3个点
            if std_dev > 0 and abs(value - mean) > 3 * std_dev:
                return True
        
        return False
    
    def _is_increasing_trend(self, values: List[float]) -> bool:
        """判断是否为递增趋势"""
        if len(values) < 3:
            return False
        
        # 简单线性回归计算趋势
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return False
        
        slope = numerator / denominator
        return slope > 0.01  # 阈值可调整
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """计算增长率"""
        if len(values) < 2:
            return 0.0
        
        first_value = values[0]
        last_value = values[-1]
        
        if first_value == 0:
            return 0.0
        
        return (last_value - first_value) / first_value

# 使用示例
async def demo_kubernetes_monitoring():
    """演示Kubernetes监控"""
    print("\n=== Kubernetes监控体系演示 ===")
    
    monitoring = KubernetesMonitoring()
    
    # 显示监控栈组件
    print("监控栈组件:")
    for component, setup_func in monitoring.monitoring_stack.items():
        component_info = setup_func()
        print(f"\n{component.upper()}:")
        print(f"  目的: {component_info['purpose']}")
        
        if 'metrics_collection' in component_info:
            print("  收集指标:")
            for category, metrics in component_info['metrics_collection'].items():
                print(f"    {category}:")
                for metric in metrics[:3]:  # 只显示前3个指标
                    print(f"      - {metric}")
                if len(metrics) > 3:
                    print(f"      ... 还有{len(metrics)-3}个指标")
    
    # 监控指标分析演示
    print("\n\n=== 监控指标分析演示 ===")
    analyzer = MonitoringMetricsAnalyzer()
    
    # 模拟监控数据
    sample_metrics = {
        'node_cpu': {'node1': 75.5, 'node2': 82.3, 'node3': 45.2},
        'pod_cpu': {'pod-a': 85.2, 'pod-b': 92.1, 'pod-c': 23.4},
        'node_memory': {'node1': 1024, 'node2': 2048, 'node3': 512},
        'pod_memory': {'pod-a': 512, 'pod-b': 1024, 'pod-c': 128}
    }
    
    # 资源使用分析
    resource_analysis = analyzer._analyze_resource_usage(sample_metrics)
    print("CPU热点分析:")
    for hotspot in resource_analysis['cpu_analysis'].get('pod_cpu_hotspots', []):
        print(f"  Pod {hotspot['pod']}: CPU使用率 {hotspot['cpu_usage']:.1f}% (严重程度: {hotspot['severity']})")
    
    print("\n内存使用分析:")
    for pod, usage in sample_metrics['pod_memory'].items():
        print(f"  Pod {pod}: {usage}MB 内存使用")

# 运行演示
import asyncio
asyncio.run(demo_kubernetes_monitoring())
```

## 日志管理

### ELK Stack + Fluentd日志方案
```python
class LoggingManagement:
    """日志管理解决方案"""
    
    def __init__(self):
        self.logging_stack = {
            'fluentd': self._setup_fluentd,
            'elasticsearch': self._setup_elasticsearch,
            'kibana': self._setup_kibana,
            'log_pipeline': self._setup_log_pipeline
        }
    
    def _setup_fluentd(self):
        """Fluentd配置"""
        return {
            'purpose': '统一日志收集和转发',
            'deployment_manifest': '''
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fluentd
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fluentd
rules:
- apiGroups: [""]
  resources:
  - pods
  - namespaces
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: fluentd
roleRef:
  kind: ClusterRole
  name: fluentd
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: fluentd
  namespace: kube-system
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
  labels:
    app: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      tolerations:
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1.16.4-debian-elasticsearch8-1
        env:
          - name: FLUENTD_ARGS
            value: --no-supervisor -q
        volumeMounts:
        - name: varlog
          mountPath: /var/log
          readOnly: true
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: config-volume
          mountPath: /etc/fluent/config.d
          readOnly: true
        resources:
          limits:
            memory: 200Mi
          requests:
            cpu: 100m
            memory: 200Mi
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: config-volume
        configMap:
          name: fluentd-config
            ''',
            'fluentd_config': '''
<source>
  @type tail
  @id kubernetes-containers
  path /var/log/containers/*.log
  pos_file /var/log/fluentd-containers.log.pos
  tag kubernetes.*
  read_from_head true
  <parse>
    @type json
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

<filter kubernetes.**>
  @type kubernetes_metadata
  @id filter_kube_metadata
</filter>

<filter kubernetes.**>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    env "#{ENV['NODE_NAME']}"
  </record>
</filter>

<match kubernetes.**>
  @type elasticsearch
  @id output-elasticsearch
  host elasticsearch.monitoring.svc.cluster.local
  port 9200
  index_name kubernetes
  type_name _doc
  include_tag_key true
  tag_key @log_name
  logstash_format true
  logstash_prefix kubernetes
  logstash_dateformat %Y.%m.%d
  <buffer>
    @type file
    path /var/log/fluentd-buffers/kubernetes.system.buffer
    flush_mode interval
    retry_type exponential_backoff
    flush_thread_count 2
    flush_interval 5s
    retry_forever
    retry_max_interval 30
    chunk_limit_size 2M
    total_limit_size 500M
    overflow_action block
  </buffer>
</match>
            '''
        }
    
    def _setup_elasticsearch(self):
        """Elasticsearch配置"""
        return {
            'purpose': '分布式搜索和分析引擎',
            'deployment_manifest': '''
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
  namespace: monitoring
spec:
  serviceName: elasticsearch
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        ports:
        - containerPort: 9200
          name: rest
        - containerPort: 9300
          name: inter-node
        env:
        - name: cluster.name
          value: kubernetes-logging
        - name: node.name
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: discovery.seed_hosts
          value: "elasticsearch-0.elasticsearch,elasticsearch-1.elasticsearch,elasticsearch-2.elasticsearch"
        - name: cluster.initial_master_nodes
          value: "elasticsearch-0,elasticsearch-1,elasticsearch-2"
        - name: ES_JAVA_OPTS
          value: "-Xms1g -Xmx1g"
        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            cpu: 1
            memory: 2Gi
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: elasticsearch
  namespace: monitoring
spec:
  clusterIP: None
  selector:
    app: elasticsearch
  ports:
  - port: 9200
    name: rest
  - port: 9300
    name: inter-node
            ''',
            'index_lifecycle_policy': '''
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: elasticsearch
spec:
  version: 8.11.0
  nodeSets:
  - name: default
    count: 3
    config:
      node.store.allow_mmap: false
      xpack.security.enabled: false
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          resources:
            requests:
              memory: 2Gi
              cpu: 1
            limits:
              memory: 2Gi
              cpu: 1
    volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 10Gi
  - name: master
    count: 3
    config:
      node.roles: ["master"]
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          resources:
            requests:
              memory: 512Mi
              cpu: 500m
            limits:
              memory: 512Mi
              cpu: 500m
            '''
        }

class LogAnalysis:
    """日志分析工具"""
    
    def __init__(self):
        self.analyzers = {
            'error_analyzer': self._analyze_errors,
            'performance_analyzer': self._analyze_performance_logs,
            'security_analyzer': self._analyze_security_logs,
            'trend_analyzer': self._analyze_log_trends
        }
    
    def _analyze_errors(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析错误日志"""
        error_analysis = {
            'error_summary': {
                'total_errors': 0,
                'error_types': {},
                'error_sources': {},
                'error_trends': {}
            },
            'critical_errors': [],
            'patterns': [],
            'recommendations': []
        }
        
        # 简化的错误分析逻辑
        error_patterns = [
            'ERROR',
            'Exception',
            'FATAL',
            'Panic',
            'OutOfMemory',
            'ConnectionRefused',
            'Timeout'
        ]
        
        for log_entry in log_data:
            message = log_entry.get('message', '').upper()
            
            # 统计错误类型
            for pattern in error_patterns:
                if pattern in message:
                    error_analysis['error_summary']['total_errors'] += 1
                    
                    error_type = error_analysis['error_summary']['error_types'].get(pattern, 0)
                    error_analysis['error_summary']['error_types'][pattern] = error_type + 1
                    
                    # 记录错误来源
                    source = log_entry.get('pod', 'unknown')
                    source_count = error_analysis['error_summary']['error_sources'].get(source, 0)
                    error_analysis['error_summary']['error_sources'][source] = source_count + 1
                    
                    # 识别严重错误
                    if pattern in ['FATAL', 'Panic', 'OutOfMemory']:
                        error_analysis['critical_errors'].append({
                            'timestamp': log_entry.get('timestamp'),
                            'source': source,
                            'pattern': pattern,
                            'message': log_entry.get('message', '')[:200]
                        })
        
        return error_analysis
    
    def _analyze_performance_logs(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能相关日志"""
        performance_analysis = {
            'latency_issues': [],
            'resource_issues': [],
            'timeout_issues': [],
            'patterns': []
        }
        
        latency_patterns = ['slow', 'timeout', 'took too long']
        resource_patterns = ['memory', 'cpu', 'disk', 'resource']
        
        for log_entry in log_data:
            message = log_entry.get('message', '').lower()
            source = log_entry.get('pod', 'unknown')
            
            # 分析延迟问题
            for pattern in latency_patterns:
                if pattern in message:
                    performance_analysis['latency_issues'].append({
                        'source': source,
                        'pattern': pattern,
                        'message': log_entry.get('message', '')[:200],
                        'timestamp': log_entry.get('timestamp')
                    })
            
            # 分析资源问题
            for pattern in resource_patterns:
                if pattern in message:
                    performance_analysis['resource_issues'].append({
                        'source': source,
                        'pattern': pattern,
                        'message': log_entry.get('message', '')[:200],
                        'timestamp': log_entry.get('timestamp')
                    })
        
        return performance_analysis
    
    def _analyze_security_logs(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析安全相关日志"""
        security_analysis = {
            'failed_attempts': [],
            'suspicious_activities': [],
            'authentication_issues': [],
            'threat_indicators': []
        }
        
        security_patterns = {
            'failed_attempts': ['failed', 'unauthorized', 'access denied', 'authentication failed'],
            'suspicious': ['suspicious', 'malicious', 'attack', 'intrusion'],
            'authentication': ['login failed', 'invalid token', 'expired', 'unauthorized']
        }
        
        for log_entry in log_data:
            message = log_entry.get('message', '').lower()
            
            for category, patterns in security_patterns.items():
                for pattern in patterns:
                    if pattern in message:
                        if category == 'failed_attempts':
                            security_analysis['failed_attempts'].append(log_entry)
                        elif category == 'suspicious':
                            security_analysis['suspicious_activities'].append(log_entry)
                        elif category == 'authentication':
                            security_analysis['authentication_issues'].append(log_entry)
        
        return security_analysis
    
    def _analyze_log_trends(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析日志趋势"""
        trend_analysis = {
            'hourly_patterns': {},
            'daily_patterns': {},
            'error_trends': {},
            'volume_trends': {}
        }
        
        # 按小时统计
        for log_entry in log_data:
            timestamp = log_entry.get('timestamp')
            if timestamp:
                hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
                trend_analysis['hourly_patterns'][hour] = trend_analysis['hourly_patterns'].get(hour, 0) + 1
        
        return trend_analysis

# 使用示例
async def demo_logging_management():
    """演示日志管理"""
    print("\n=== Kubernetes日志管理演示 ===")
    
    logging_mgmt = LoggingManagement()
    
    # 显示日志栈组件
    for component, setup_func in logging_mgmt.logging_stack.items():
        component_info = setup_func()
        print(f"\n{component.upper()}:")
        print(f"  目的: {component_info['purpose']}")
    
    # 日志分析演示
    print("\n\n=== 日志分析演示 ===")
    log_analyzer = LogAnalysis()
    
    # 模拟日志数据
    sample_logs = [
        {
            'timestamp': '2024-01-15T10:30:00Z',
            'pod': 'web-app-123',
            'level': 'ERROR',
            'message': 'Database connection timeout occurred'
        },
        {
            'timestamp': '2024-01-15T10:31:00Z',
            'pod': 'web-app-124',
            'level': 'WARN',
            'message': 'High memory usage detected'
        },
        {
            'timestamp': '2024-01-15T10:32:00Z',
            'pod': 'web-app-125',
            'level': 'INFO',
            'message': 'Request processed successfully'
        }
    ]
    
    # 错误分析
    error_analysis = log_analyzer._analyze_errors(sample_logs)
    print("错误分析结果:")
    print(f"  总错误数: {error_analysis['error_summary']['total_errors']}")
    print(f"  错误类型分布: {error_analysis['error_summary']['error_types']}")
    print(f"  错误来源分布: {error_analysis['error_summary']['error_sources']}")

# 运行演示
import asyncio
asyncio.run(demo_logging_management())
```

这个Kubernetes运维监控与故障排除文档详细介绍了：

1. **监控体系** - Prometheus + Grafana监控方案的完整配置和实现
2. **日志管理** - ELK Stack + Fluentd日志收集和处理方案
3. **监控指标分析器** - 智能分析资源使用、性能、异常检测和容量规划
4. **日志分析工具** - 自动分析错误、性能、安全和趋势

通过这些内容，可以构建完整的Kubernetes运维监控体系，实现：
- 实时监控集群状态和资源使用情况
- 智能异常检测和告警
- 全面的日志收集和分析
- 主动的容量规划和性能优化

这样可以确保Kubernetes集群的稳定运行和高效运维。