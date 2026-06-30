# TensorRT Plugins

## Contents

| Plugin                                      | 名称                           | Versions         |
| ------------------------------------------- | ------------------------------ | ---------------- |
| bertQKVToContextPlugin [DEPRECATED]         | 自定义QKV转上下文动态插件      | 1, 2, 3, 4, 5, 6 |
| cropAndResizePlugin                         | 裁剪并调整大小动态插件         | 2                |
| decodeBbox3DPlugin [DEPRECATED]             | 解码3D边界框插件               | 1                |
| detectionLayerPlugin [DEPRECATED]           | 检测层插件                     | 1                |
| disentangledAttentionPlugin [DEPRECATED]    | 解耦注意力插件                 | 1                |
| disentangledAttentionPlugin                 | 解耦注意力插件                 | 2                |
| efficientNMSPlugin [DEPRECATED]             | 高效NMS插件                    | 1                |
| efficientNMSPlugin/tftrt [DEPRECATED]       | 显式TF高效NMS插件              | 1                |
| efficientNMSPlugin/tftrt [DEPRECATED]       | 隐式TF高效NMS插件              | 1                |
| embLayerNormPlugin [DEPRECATED]             | 自定义嵌入层归一化动态插件     | 1, 2, 3          |
| embLayerNormPlugin                          | 自定义嵌入层归一化动态插件     | 4, 5, 6          |
| fcPlugin [DEPRECATED]                       | 自定义全连接动态插件           | 1                |
| flattenConcat [DEPRECATED]                  | 展平拼接插件                   | 1                |
| generateDetectionPlugin [DEPRECATED]        | 生成检测结果插件               | 1                |
| gridAnchorPlugin [DEPRECATED]               | 网格锚点插件                   | 1                |
| gridAnchorPlugin [DEPRECATED]               | 矩形网格锚点插件               | 1                |
| groupNormalizationPlugin [DEPRECATED]       | 组归一化插件                   | 1                |
| instanceNormalizationPlugin [DEPRECATED]    | 实例归一化插件                 | 1, 2             |
| instanceNormalizationPlugin                 | 实例归一化插件                 | 3                |
| modulatedDeformConvPlugin [DEPRECATED]      | 调制可变形卷积插件             | 1                |
| modulatedDeformConvPlugin                   | 调制可变形卷积插件             | 2                |
| multilevelCropAndResizePlugin [DEPRECATED]  | 多级裁剪并调整大小插件         | 1                |
| multilevelProposeROI [DEPRECATED]           | 多级提议ROI插件                | 1                |
| multiscaleDeformableAttnPlugin [DEPRECATED] | 多尺度可变形注意力插件         | 1                |
| multiscaleDeformableAttnPlugin              | 多尺度可变形注意力插件         | 2                |
| nvFasterRCNN [DEPRECATED]                   | RPROI插件                      | 1                |
| pillarScatterPlugin [DEPRECATED]            | 柱状散射插件                   | 1                |
| priorBoxPlugin [DEPRECATED]                 | 先验框插件                     | 1                |
| proposalLayerPlugin [DEPRECATED]            | 提议层插件                     | 1                |
| pyramidROIAlignPlugin [DEPRECATED]          | 金字塔ROI对齐插件              | 1                |
| regionPlugin [DEPRECATED]                   | 区域插件                       | 1                |
| reorgPlugin [DEPRECATED]                    | 重组插件                       | 2                |
| roiAlignPlugin [DEPRECATED]                 | ROI对齐插件                    | 1                |
| roiAlignPlugin                              | ROI对齐插件                    | 2                |
| resizeNearestPlugin [DEPRECATED]            | 最近邻缩放插件                 | 1                |
| scatterElementsPlugin [DEPRECATED]          | 元素散布插件                   | 1                |
| scatterElementsPlugin                       | 元素散布插件                   | 2                |
| scatterPlugin [DEPRECATED]                  | 多维散布插件                   | 1                |
| skipLayerNormPlugin [DEPRECATED]            | 自定义跳跃连接层归一化动态插件 | 1, 2, 3, 4       |
| skipLayerNormPlugin                         | 自定义跳跃连接层归一化动态插件 | 5, 6, 7, 8       |
| topkLastDimPlugin                           | 末维TopK插件                   | 1                |
| voxelGeneratorPlugin [DEPRECATED]           | 体素生成插件                   | 1                |



## Known Limitations

  - None
