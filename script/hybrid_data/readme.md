在处理数据的时候应该注意：
（1）所有的基因都有表观遗传的数据以及空间结构数据
（2）但是不是所有的基因都有标签 （没有测到）

因此：

（1） graph中应该包含所有需要的节点信息和标签信息
（2） 只有标签信息是需要mask的！！ 也就是说做mask需要注意，只读取有label的部分


操作步骤

# 图模型相关

（2）construct_hic_map.sh
# 特征及标签相关
(1) bash split_gene_expression.sh
(2) bash convert_histone_wrapper.sh
(3) bash split_histone.sh
(4) bash extract_histone_feature.sh