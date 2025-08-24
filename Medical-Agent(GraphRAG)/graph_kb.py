import networkx as nx
import pickle
import os
from typing import List, Tuple, Dict, Optional, Set
from pyvis.network import Network
from graph_schema import ENTITY_TYPES, RELATION_TYPES, ENTITY_COLORS  # 引入更新后的schema

class GraphKnowledgeBase:
    """基于NetworkX的医疗知识图谱实现，适配更新后的实体和关系类型"""
    
    def __init__(self):
        """初始化知识图谱（有向图）"""
        self.graph = nx.DiGraph()
        # 从schema导入类型描述
        self.entity_type_descriptions = {k: v.split("（")[0] for k, v in ENTITY_TYPES.items()}
        self.relation_type_descriptions = {k: v.split("（")[0] for k, v in RELATION_TYPES.items()}

    def add_relation(self, source: str, source_type: str, target: str, target_type: str, relation_type: str) -> bool:
        """添加实体关系（验证类型是否在schema定义范围内）"""
        # 验证实体类型和关系类型是否合法
        if source_type not in ENTITY_TYPES:
            print(f"警告：源实体类型 {source_type} 不在定义的ENTITY_TYPES中")
            return False
        if target_type not in ENTITY_TYPES:
            print(f"警告：目标实体类型 {target_type} 不在定义的ENTITY_TYPES中")
            return False
        if relation_type not in RELATION_TYPES:
            print(f"警告：关系类型 {relation_type} 不在定义的RELATION_TYPES中")
            return False

        try:
            # 添加源实体（如果不存在）
            if source not in self.graph.nodes:
                self.graph.add_node(
                    source, 
                    type=source_type,
                    label=self.entity_type_descriptions[source_type]
                )
            
            # 添加目标实体（如果不存在）
            if target not in self.graph.nodes:
                self.graph.add_node(
                    target, 
                    type=target_type,
                    label=self.entity_type_descriptions[target_type]
                )
            
            # 添加关系
            self.graph.add_edge(
                source, 
                target, 
                type=relation_type,
                label=self.relation_type_descriptions[relation_type]
            )
            return True
        except Exception as e:
            print(f"添加关系失败: {str(e)}")
            return False

    def query_related_entities(self, entity: str, relation: Optional[str] = None, max_hops: int = 1) -> List[Tuple[str, str, str, str]]:
        """查询实体的相关实体（支持新增的RiskFactor等类型）"""
        if entity not in self.graph.nodes:
            return []
        
        result = []
        visited = set()
        queue = [(entity, 0)]  # (当前实体, 当前跳数)
        
        while queue:
            current_entity, hops = queue.pop(0)
            
            if current_entity in visited or hops > max_hops:
                continue
                
            visited.add(current_entity)
            
            # 遍历当前实体的所有出边
            for neighbor in self.graph.successors(current_entity):
                edge_data = self.graph.get_edge_data(current_entity, neighbor)
                rel_type = edge_data.get('type', '')
                
                # 如果指定了关系类型，则只保留匹配的关系（支持DIAGNOSES、PREVENTS等）
                if relation and rel_type != relation:
                    continue
                
                # 获取邻居实体类型
                neighbor_type = self.graph.nodes[neighbor].get('type', '')
                
                # 添加到结果
                result.append((current_entity, rel_type, neighbor, neighbor_type))
                
                # 如果未达到最大跳数，继续遍历
                if hops < max_hops:
                    queue.append((neighbor, hops + 1))
        
        return result

    def visualize(self, filename: str = "knowledge_graph.html", 
                 highlight_entities: List[str] = None, 
                 max_nodes: int = 100) -> None:
        """可视化知识图谱（使用ENTITY_COLORS定义的颜色）"""
        # 创建可视化网络
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#f8f9fa", 
            font_color="black",
            directed=True
        )
        
        # 从schema导入颜色映射（替换原硬编码颜色）
        color_map = ENTITY_COLORS
        
        # 限制节点数量，避免可视化过于复杂
        nodes = list(self.graph.nodes)[:max_nodes]
        
        # 添加节点
        for node in nodes:
            node_type = self.graph.nodes[node].get('type', 'Unknown')
            # 使用schema中定义的颜色，未知类型用灰色
            color = color_map.get(node_type, "#CCCCCC")
            
            # 如果是需要高亮的实体，使用更亮的颜色
            if highlight_entities and node in highlight_entities:
                color = self._lighten_color(color, 30)
            
            net.add_node(
                node, 
                label=node, 
                title=f"{node}（{ENTITY_TYPES.get(node_type, node_type)}）",  # 显示完整类型描述
                color=color,
                size=25 if (highlight_entities and node in highlight_entities) else 20
            )
        
        # 添加边（支持新增的关系类型）
        for u, v, data in self.graph.edges(data=True):
            if u in nodes and v in nodes:
                rel_type = data.get('type', '')
                net.add_edge(
                    u, 
                    v, 
                    label=self.relation_type_descriptions.get(rel_type, rel_type),
                    color="#AAAAAA",
                    width=2
                )
        
        # 配置物理引擎
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "springLength": 100
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "timestep": 0.35
            }
        }
        """)
        
        # 保存可视化结果
        net.write_html(filename)
        print(f"知识图谱可视化已保存到 {filename}")

    # 其他方法（find_paths、get_entity_type、save_to_file等）保持不变
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[Tuple[str, str, str]]]:
        if source not in self.graph.nodes or target not in self.graph.nodes:
            return []
        
        try:
            paths = []
            for path in nx.all_simple_paths(self.graph, source=source, target=target, cutoff=max_length):
                relation_path = []
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    rel_type = self.graph.get_edge_data(u, v).get('type', '')
                    relation_path.append((u, rel_type, v))
                if relation_path:
                    paths.append(relation_path)
            return paths
        except Exception as e:
            print(f"查找路径失败: {str(e)}")
            return []

    def get_entity_type(self, entity: str) -> Optional[str]:
        if entity in self.graph.nodes:
            return self.graph.nodes[entity].get('type')
        return None

    def get_all_entities(self, entity_type: Optional[str] = None) -> List[str]:
        if entity_type:
            return [node for node, data in self.graph.nodes(data=True) if data.get('type') == entity_type]
        return list(self.graph.nodes)

    def save_to_file(self, filename: str) -> bool:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"知识图谱已保存到 {filename}")
            return True
        except Exception as e:
            print(f"保存知识图谱失败: {str(e)}")
            return False

    def load_from_file(self, filename: str) -> bool:
        try:
            if not os.path.exists(filename):
                print(f"知识图谱文件 {filename} 不存在")
                return False
                
            with open(filename, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"已从 {filename} 加载知识图谱，包含 {len(self.graph.nodes)} 个实体和 {len(self.graph.edges)} 个关系")
            return True
        except Exception as e:
            print(f"加载知识图谱失败: {str(e)}")
            return False

    def _lighten_color(self, color: str, percent: int) -> str:
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        import numpy as np
        
        rgb = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
        hsv = rgb_to_hsv(rgb)
        hsv[2] = min(1.0, hsv[2] + percent / 100.0)
        rgb = hsv_to_rgb(hsv)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
