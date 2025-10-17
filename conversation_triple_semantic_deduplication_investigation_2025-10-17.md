# KAG Triple语义去重调研对话记录

**日期**: 2025-10-17  
**主题**: 调研KAG仓库中是否实现了对提取的triple进行语义去重

---

## 用户提问

这个repo中是否涉及到对提取的triple进行语义去重？而不是字符去重

---

## 调研过程

### 1. 初步探索

首先探索了仓库结构，搜索了包含"triple"、"去重/dedup"等关键词的文件：

- 找到了61个包含"triple"的文件
- 找到了10个包含"去重/dedup/deduplicate"的文件
- 没有找到直接包含"semantic.*dedup"或"语义.*去重"的文件

### 2. 核心代码分析

重点分析了以下文件：

#### 2.1 `kag/builder/component/extractor/schema_free_extractor.py`

**Entity去重机制**（第119-150行）：
```python
def _named_entity_recognition_process(self, passage, ner_result):
    if self.external_graph:
        extra_ner_result = self.external_graph.ner(passage)
    else:
        extra_ner_result = []
    output = []
    dedup = set()  # 使用简单的set进行字符去重
    for item in extra_ner_result:
        name = item.name
        label = item.label
        spg_type = self.schema.get(label)
        if spg_type is None:
            label = "Others"
            item.label = label
        description = item.properties.get("desc", "")
        semantic_type = item.properties.get("semanticType", label)
        if name not in dedup:  # 基于name字符串判断
            dedup.add(name)
            output.append({
                "name": name,
                "type": semantic_type,
                "category": label,
                "description": description,
            })
    for item in ner_result:
        name = item.get("name", None)
        if name and name not in dedup:  # 同样是字符判断
            dedup.add(name)
            output.append(item)
    return output
```

**Triple组装机制**（第318-362行）：
```python
@staticmethod
def assemble_sub_graph_with_triples(
    sub_graph: SubGraph, entities: List[Dict], triples: List[list]
):
    """
    Assembles edges in the subgraph based on a list of triples and entities.
    Args:
        sub_graph (SubGraph): The subgraph to add edges to.
        entities (List[Dict]): A list of entities, for looking up category information.
        triples (List[list]): A list of triples, each representing a relationship to be added to the subgraph.
    Returns:
        The constructed subgraph.
    """

    def get_category_and_name(entities_data, entity_name):
        for entity in entities_data:
            if processing_phrases(entity["name"]) == processing_phrases(
                entity_name
            ):
                return entity["category"], entity["name"]
        return None, None

    for tri in triples:
        if tri is None or len(tri) != 3:
            continue
        s_category, s_name = get_category_and_name(entities, tri[0])
        tri[0] = processing_phrases(tri[0])
        if tri[0] == "":
            continue
        if s_category is None:
            s_category = OTHER_TYPE
            s_name = tri[0]
            sub_graph.add_node(s_name, s_name, s_category)
        o_category, o_name = get_category_and_name(entities, tri[2])
        if o_name == "":
            continue
        if o_category is None:
            o_name = processing_phrases(tri[2])
            o_category = OTHER_TYPE
            sub_graph.add_node(o_name, o_name, o_category)
        edge_type = to_camel_case(tri[1])
        if edge_type:
            sub_graph.add_edge(s_name, s_category, edge_type, o_name, o_category)  # 直接添加，无去重

    return sub_graph
```

**关键发现**：Triple直接被添加到图中，没有任何去重逻辑。

#### 2.2 `kag/builder/component/extractor/knowledge_unit_extractor.py`

类似的结构，在`assemble_sub_graph_with_triples`方法（第427-482行）中也是直接添加triple：

```python
@staticmethod
def assemble_sub_graph_with_triples(
    sub_graph: SubGraph, entities: List[Dict], triples: List[list]
):
    # ...
    for tri in triples:
        if len(tri) != 4:
            continue
        # ... 处理逻辑 ...
        if edge_type:
            sub_graph.add_edge(
                s_name,
                s_category,
                edge_type,
                o_name,
                o_category,
                properties=properties,
            )  # 直接添加，无去重
    return sub_graph
```

#### 2.3 `kag/builder/component/extractor/schema_constraint_extractor.py`

同样没有triple去重机制（第388-414行）：

```python
@staticmethod
def add_relations_to_graph(
    sub_graph: SubGraph, entities: List[Dict], relations: List[list]
):
    """
    Add edges to the subgraph based on a list of relations and entities.
    """
    for rel in relations:
        if len(rel) != 5:
            continue
        s_name, s_category, predicate, o_name, o_category = rel
        s_name = processing_phrases(s_name)
        sub_graph.add_node(s_name, s_name, s_category)
        o_name = processing_phrases(o_name)
        sub_graph.add_node(o_name, o_name, o_category)
        edge_type = to_camel_case(predicate)
        if edge_type:
            sub_graph.add_edge(s_name, s_category, edge_type, o_name, o_category)  # 直接添加
    return sub_graph
```

### 3. 语义相似度基础设施分析

#### 3.1 `kag/common/text_sim_by_vector.py`

仓库中确实存在基于向量embedding的语义相似度计算能力：

```python
def cosine_similarity(vector1, vector2):
    if vector1.shape == (0,) or vector2.shape == (0,):
        return 0.0
    cosine = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return cosine

class TextSimilarity:
    def __init__(self, vectorizer: Vectorizer = None, **kwargs):
        # 初始化向量化模型
        self.vectorize_model = vectorizer
        self.cached_embs = {}

    def text_sim_result(
        self, mention, candidates: List[str], topk=1, low_score=0.63, is_cached=False
    ):
        """
        output: [(candi_name, candi_score),...]
        """
        # 计算mention与候选文本的余弦相似度
        # ...
```

#### 3.2 `kag/builder/component/postprocessor/kag_postprocessor.py`

后处理器使用向量相似度进行**Entity Linking**，但不涉及triple：

```python
@retry(stop=stop_after_attempt(3), reraise=True)
def _entity_link(
    self, graph: SubGraph, property_key: str = "name", labels: List[str] = None
):
    """
    Performs entity linking based on the given property key and labels.
    """
    vector_field_name = get_vector_field_name(property_key)
    for node in graph.nodes:
        # ... 通过向量相似度进行entity linking
        vector = node.properties.get(vector_field_name)
        if vector:
            # 使用向量搜索找到相似实体
            similar_nodes = self._search_client.search_vector(
                label=label,
                property_key=property_key,
                query_vector=[float(x) for x in vector],
                topk=1,
                params={},
            )
            # 添加相似边
            for item in all_similar_nodes:
                score = item["score"]
                if score >= self.similarity_threshold and node.id != item["node"]["id"]:
                    graph.add_edge(
                        node.id,
                        node.label,
                        KAGConstants.KAG_SIMILAR_EDGE_NAME,
                        item["node"]["id"],
                        item["node"]["__labels__"][0],
                    )
```

**关键发现**：语义相似度功能仅用于entity的链接，不用于triple去重。

### 4. SubGraph数据结构分析

#### `kag/interface/common/model/sub_graph.py`

查看了SubGraph的核心数据结构：

```python
class SubGraph(object):
    id: str
    nodes: List[Node] = list()
    edges: List[Edge] = list()

    def add_node(self, id: str, name: str, label: str, properties=None):
        if not properties:
            properties = dict()
        store_node = self.get_node_by_id(id, label)
        if not store_node:
            self.nodes.append(
                Node(_id=id, name=name, label=label, properties=properties)
            )
            return self
        if store_node and properties is not None:
            update_prop = dict(properties)
            update_prop.update(store_node.properties if store_node.properties else {})
            store_node.properties = update_prop
        return self

    def add_edge(
        self, s_id: str, s_label: str, p: str, o_id: str, o_label: str, properties=None
    ):
        if not properties:
            properties = dict()
        s_node = Node(_id=s_id, name=s_id, label=s_label, properties={})
        o_node = Node(_id=o_id, name=o_id, label=o_label, properties={})
        self.edges.append(
            Edge(
                _id="", from_node=s_node, to_node=o_node, label=p, properties=properties
            )
        )  # 直接append，没有去重检查
        return self
```

**关键发现**：`add_edge`方法直接append边，没有任何去重逻辑。

### 5. 分支名称观察

当前分支名称：`cursor/check-for-semantic-triple-deduplication-af2c`

这表明可能正是在调研是否需要添加triple语义去重功能。

---

## 调研结论

### 明确答案

**这个仓库中目前没有对提取的triple进行语义去重，也没有进行字符去重。**

### 详细分析

#### 1. **Entity层面的去重**
- ✅ **有字符去重**：使用简单的`set()`基于entity的name字符串进行去重
- ❌ **无语义去重**：不同名称但语义相同的entity不会被去重

示例：
```python
dedup = set()
if name not in dedup:  # 仅基于字符串完全匹配
    dedup.add(name)
    output.append(item)
```

#### 2. **Triple层面的去重**
- ❌ **无字符去重**：相同的triple可能被多次添加
- ❌ **无语义去重**：语义相同但表述不同的triple不会被去重

所有extractor的`assemble_sub_graph_with_triples`方法都是直接将triple添加到edges列表，没有任何去重逻辑：
```python
for tri in triples:
    # ... 处理 ...
    sub_graph.add_edge(...)  # 直接添加
```

#### 3. **语义相似度能力现状**

仓库已具备语义相似度计算的基础设施：

**已有能力**：
- `TextSimilarity`类：提供基于向量embedding的余弦相似度计算
- `BatchVectorizer`：批量生成节点属性的embedding向量
- `KAGPostProcessor`：使用向量相似度进行entity linking

**使用范围**：
- ✅ Entity的相似度链接（entity linking）
- ✅ 文本检索和召回
- ❌ Triple的去重（**未使用**）

#### 4. **潜在影响**

没有triple去重可能导致：
1. **数据冗余**：相同或相似的关系被重复存储
2. **图结构膨胀**：大量重复边导致图数据库性能下降
3. **查询效率降低**：检索时需要处理重复关系
4. **语义混淆**：如"出生于"和"born in"被视为不同关系

---

## 改进建议

如果需要实现triple的语义去重，可以考虑以下方案：

### 方案1：在提取阶段进行去重

在`assemble_sub_graph_with_triples`方法中添加去重逻辑：

```python
@staticmethod
def assemble_sub_graph_with_triples_with_dedup(
    sub_graph: SubGraph, 
    entities: List[Dict], 
    triples: List[list],
    similarity_threshold: float = 0.85
):
    # 1. 字符去重
    unique_triples = {}
    for tri in triples:
        key = f"{tri[0]}|{tri[1]}|{tri[2]}"
        if key not in unique_triples:
            unique_triples[key] = tri
    
    # 2. 语义去重（使用TextSimilarity）
    text_sim = TextSimilarity()
    deduplicated_triples = []
    
    for tri in unique_triples.values():
        triple_text = f"{tri[0]} {tri[1]} {tri[2]}"
        is_duplicate = False
        
        for existing_tri in deduplicated_triples:
            existing_text = f"{existing_tri[0]} {existing_tri[1]} {existing_tri[2]}"
            sim_score = text_sim.text_sim_result(triple_text, [existing_text], topk=1)
            
            if sim_score and sim_score[0][1] >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated_triples.append(tri)
    
    # 3. 添加到子图
    for tri in deduplicated_triples:
        # ... 原有的添加逻辑 ...
```

### 方案2：在后处理阶段进行去重

在`KAGPostProcessor`中添加triple去重功能：

```python
class KAGPostProcessor(PostProcessorABC):
    def semantic_triple_dedup(self, graph: SubGraph, threshold: float = 0.85):
        """对triple进行语义去重"""
        text_sim = TextSimilarity(self.vectorize_model)
        
        deduplicated_edges = []
        edge_texts = []
        
        for edge in graph.edges:
            edge_text = f"{edge.from_id} {edge.label} {edge.to_id}"
            
            # 检查是否与已有边语义重复
            if edge_texts:
                sim_results = text_sim.text_sim_result(
                    edge_text, edge_texts, topk=1, low_score=threshold
                )
                if not sim_results:  # 不重复
                    deduplicated_edges.append(edge)
                    edge_texts.append(edge_text)
            else:
                deduplicated_edges.append(edge)
                edge_texts.append(edge_text)
        
        graph.edges = deduplicated_edges
        return graph
```

### 方案3：定义相似度阈值配置

在配置文件中添加triple去重相关配置：

```yaml
# kag_config.yaml
post_processor:
  type: "kag_post_processor"
  similarity_threshold: 0.9
  enable_triple_dedup: true  # 新增
  triple_dedup_threshold: 0.85  # 新增
  triple_dedup_method: "semantic"  # "semantic" or "exact"
```

### 关键考虑因素

1. **性能开销**：语义去重需要计算大量embedding和相似度，可能影响性能
2. **阈值设定**：需要根据实际场景调整相似度阈值（0.85-0.95范围）
3. **关系规范化**：考虑通过schema定义标准关系类型，减少同义关系
4. **批处理优化**：利用向量化批量计算，提高去重效率

---

## 附录：相关代码文件清单

### 核心提取器
- `kag/builder/component/extractor/schema_free_extractor.py`
- `kag/builder/component/extractor/knowledge_unit_extractor.py`
- `kag/builder/component/extractor/schema_constraint_extractor.py`

### 后处理
- `kag/builder/component/postprocessor/kag_postprocessor.py`

### 语义相似度
- `kag/common/text_sim_by_vector.py`
- `kag/builder/component/vectorizer/batch_vectorizer.py`

### 数据模型
- `kag/interface/common/model/sub_graph.py`
- `kag/builder/model/sub_graph.py`

### 向量化模型
- `kag/common/vectorize_model/local_bge_model.py`
- `kag/common/vectorize_model/openai_model.py`

---

**调研完成时间**: 2025-10-17  
**当前分支**: cursor/check-for-semantic-triple-deduplication-af2c
