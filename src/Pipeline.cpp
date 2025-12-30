module Pipeline;
using namespace RenderGraph;

PipelineGraphic::PipelineGraphic() noexcept {
  _inputAssembly = VkPipelineInputAssemblyStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = false};
  _viewportState = VkPipelineViewportStateCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                     .viewportCount = 1,
                                                     .scissorCount = 1};
  _rasterizer = VkPipelineRasterizationStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = false,
      .rasterizerDiscardEnable = false,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = false,
      .lineWidth = 1.0f};
  _multisampling = VkPipelineMultisampleStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = false};
  _blendAttachmentState = VkPipelineColorBlendAttachmentState{
      .blendEnable = true,
      .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                        VK_COLOR_COMPONENT_A_BIT};
  _colorBlending = VkPipelineColorBlendStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = false,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &_blendAttachmentState,
      .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};
  _depthStencil = VkPipelineDepthStencilStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = true,
      .depthWriteEnable = true,
      .depthCompareOp = VK_COMPARE_OP_LESS,
      .depthBoundsTestEnable = false,
      .stencilTestEnable = false,
      .front = {},             // Optional
      .back = {},              // Optional
      .minDepthBounds = 0.0f,  // Optional
      .maxDepthBounds = 1.0f,  // Optional
  };
  _dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
  _dynamicState = VkPipelineDynamicStateCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                   .dynamicStateCount = static_cast<uint32_t>(_dynamicStates.size()),
                                                   .pDynamicStates = _dynamicStates.data()};
}

void PipelineGraphic::setCullMode(VkCullModeFlags cullMode) noexcept { _rasterizer.cullMode = cullMode; }

void PipelineGraphic::setPolygonMode(VkPolygonMode polygonMode) noexcept { _rasterizer.polygonMode = polygonMode; }

void PipelineGraphic::setAlphaBlending(bool alphaBlending) noexcept {
  _blendAttachmentState.blendEnable = alphaBlending;
}

void PipelineGraphic::setTopology(VkPrimitiveTopology topology) noexcept { _inputAssembly.topology = topology; }

void PipelineGraphic::setDepthBias(bool depthBias) noexcept { _rasterizer.depthBiasEnable = depthBias; }

void PipelineGraphic::setDepthTest(bool depthTest) noexcept { _depthStencil.depthTestEnable = depthTest; }

void PipelineGraphic::setDepthWrite(bool depthWrite) noexcept { _depthStencil.depthWriteEnable = depthWrite; }

void PipelineGraphic::setDepthCompateOp(VkCompareOp depthCompareOp) noexcept {
  // we force skybox to have the biggest possible depth = 1 so we need to draw skybox if it's depth <= 1
  _depthStencil.depthCompareOp = depthCompareOp;
}

void PipelineGraphic::setColorBlendOp(VkBlendOp colorBlendOp) noexcept {
  _blendAttachmentState.colorBlendOp = colorBlendOp;
}

void PipelineGraphic::setTesselation(int patchControlPoints) noexcept {
  // according to specification: patchControlPoints must be greater than zero and less than or equal to
  // VkPhysicalDeviceLimits::maxTessellationPatchSize
  if (patchControlPoints == 0)
    _tessellationState = std::nullopt;
  else
    _tessellationState = VkPipelineTessellationStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .patchControlPoints = static_cast<uint32_t>(patchControlPoints)};
}

void PipelineGraphic::setColorAttachments(const std::vector<VkFormat>& colorAttachments) noexcept {
  _colorAttachments = colorAttachments;
}

void PipelineGraphic::setDepthAttachment(std::optional<VkFormat> depthAttachment) noexcept {
  _depthAttachment = depthAttachment;
}

const VkPipelineDynamicStateCreateInfo& PipelineGraphic::getDynamicState() const noexcept { return _dynamicState; }

const VkPipelineInputAssemblyStateCreateInfo& PipelineGraphic::getInputAssembly() const noexcept {
  return _inputAssembly;
}

const VkPipelineViewportStateCreateInfo& PipelineGraphic::getViewportState() const noexcept { return _viewportState; }

const VkPipelineRasterizationStateCreateInfo& PipelineGraphic::getRasterizer() const noexcept { return _rasterizer; }

const VkPipelineMultisampleStateCreateInfo& PipelineGraphic::getMultisampling() const noexcept {
  return _multisampling;
}

const VkPipelineColorBlendAttachmentState& PipelineGraphic::getBlendAttachmentState() const noexcept {
  return _blendAttachmentState;
}

const VkPipelineColorBlendStateCreateInfo& PipelineGraphic::getColorBlending() const noexcept { return _colorBlending; }

const VkPipelineDepthStencilStateCreateInfo& PipelineGraphic::getDepthStencil() const noexcept { return _depthStencil; }

const std::optional<VkPipelineTessellationStateCreateInfo>& PipelineGraphic::getTessellationState() const noexcept {
  return _tessellationState;
}

const std::vector<VkFormat>& PipelineGraphic::getColorAttachments() const noexcept { return _colorAttachments; }

const std::optional<VkFormat>& PipelineGraphic::getDepthAttachment() const noexcept { return _depthAttachment; }

Pipeline::Pipeline(const Device& device) noexcept : _device(&device) {}

const std::vector<std::pair<std::string, DescriptorSetLayout*>>& Pipeline::getDescriptorSetLayout() const noexcept {
  return _descriptorSetLayout;
}

const std::unordered_map<std::string, VkPushConstantRange>& Pipeline::getPushConstants() const noexcept {
  return _pushConstants;
}

const VkPipeline& Pipeline::getPipeline() const noexcept { return _pipeline; }

const VkPipelineLayout& Pipeline::getPipelineLayout() const noexcept { return _pipelineLayout; }

Pipeline::~Pipeline() {
  vkDestroyPipeline(_device->getLogicalDevice(), _pipeline, nullptr);
  vkDestroyPipelineLayout(_device->getLogicalDevice(), _pipelineLayout, nullptr);
}

void Pipeline::createGraphic(const PipelineGraphic& pipelineGraphic,
                             const std::vector<VkPipelineShaderStageCreateInfo>& shaderStages,
                             std::vector<std::pair<std::string, DescriptorSetLayout*>>& descriptorSetLayout,
                             const std::unordered_map<std::string, VkPushConstantRange>& pushConstants,
                             const VkPipelineVertexInputStateCreateInfo& vertexInputInfo) {
  _descriptorSetLayout = descriptorSetLayout;
  _pushConstants = pushConstants;

  // create pipeline layout
  std::vector<VkDescriptorSetLayout> descriptorSetLayoutRaw;
  descriptorSetLayoutRaw.reserve(_descriptorSetLayout.size());
  for (auto&& layout : _descriptorSetLayout) {
    descriptorSetLayoutRaw.push_back(layout.second->getDescriptorSetLayout());
  }
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                .setLayoutCount = static_cast<uint32_t>(descriptorSetLayoutRaw.size()),
                                                .pSetLayouts = descriptorSetLayoutRaw.data()};

  auto pushConstantsView = std::views::values(pushConstants);
  auto pushConstantsRaw = std::vector<VkPushConstantRange>{pushConstantsView.begin(), pushConstantsView.end()};
  if (pushConstants.size() > 0) {
    pipelineLayoutInfo.pPushConstantRanges = pushConstantsRaw.data();
    pipelineLayoutInfo.pushConstantRangeCount = pushConstantsRaw.size();
  }
  if (vkCreatePipelineLayout(_device->getLogicalDevice(), &pipelineLayoutInfo, nullptr, &_pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  // create pipeline
  VkPipelineRenderingCreateInfo renderingInfo = {};
  renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  auto colorAttachments = pipelineGraphic.getColorAttachments();
  renderingInfo.colorAttachmentCount = colorAttachments.size();
  renderingInfo.pColorAttachmentFormats = colorAttachments.data();
  auto depthAttachment = pipelineGraphic.getDepthAttachment();
  if (depthAttachment) renderingInfo.depthAttachmentFormat = depthAttachment.value();

  auto colorBlendingState = pipelineGraphic.getColorBlending();
  std::vector blendAttachments(colorAttachments.size(), pipelineGraphic.getBlendAttachmentState());
  colorBlendingState.attachmentCount = blendAttachments.size();
  colorBlendingState.pAttachments = blendAttachments.data();

  VkGraphicsPipelineCreateInfo pipelineInfo{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                            .pNext = &renderingInfo,
                                            .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
                                            .stageCount = static_cast<uint32_t>(shaderStages.size()),
                                            .pStages = shaderStages.data(),
                                            .pVertexInputState = &vertexInputInfo,
                                            .pInputAssemblyState = &pipelineGraphic.getInputAssembly(),
                                            .pViewportState = &pipelineGraphic.getViewportState(),
                                            .pRasterizationState = &pipelineGraphic.getRasterizer(),
                                            .pMultisampleState = &pipelineGraphic.getMultisampling(),
                                            .pDepthStencilState = &pipelineGraphic.getDepthStencil(),
                                            .pColorBlendState = &colorBlendingState,
                                            .pDynamicState = &pipelineGraphic.getDynamicState(),
                                            .layout = _pipelineLayout,
                                            .subpass = 0,
                                            .basePipelineHandle = nullptr};
  if (pipelineGraphic.getTessellationState())
    pipelineInfo.pTessellationState = &pipelineGraphic.getTessellationState().value();
  auto status = vkCreateGraphicsPipelines(_device->getLogicalDevice(), nullptr, 1, &pipelineInfo, nullptr, &_pipeline);
  if (status != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }
}

void Pipeline::createCompute(const VkPipelineShaderStageCreateInfo& shaderStage,
                             std::vector<std::pair<std::string, DescriptorSetLayout*>>& descriptorSetLayout,
                             const std::unordered_map<std::string, VkPushConstantRange>& pushConstants) {
  _descriptorSetLayout = descriptorSetLayout;
  _pushConstants = pushConstants;

  // create pipeline layout
  std::vector<VkDescriptorSetLayout> descriptorSetLayoutRaw;
  descriptorSetLayoutRaw.reserve(_descriptorSetLayout.size());
  for (auto& layout : _descriptorSetLayout) {
    descriptorSetLayoutRaw.push_back(layout.second->getDescriptorSetLayout());
  }
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = descriptorSetLayout.size();
  pipelineLayoutInfo.pSetLayouts = descriptorSetLayoutRaw.data();

  auto pushConstantsView = std::views::values(pushConstants);
  auto pushConstantsRaw = std::vector<VkPushConstantRange>{pushConstantsView.begin(), pushConstantsView.end()};
  if (pushConstants.size() > 0) {
    pipelineLayoutInfo.pPushConstantRanges = pushConstantsRaw.data();
    pipelineLayoutInfo.pushConstantRangeCount = pushConstantsRaw.size();
  }

  if (vkCreatePipelineLayout(_device->getLogicalDevice(), &pipelineLayoutInfo, nullptr, &_pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  VkComputePipelineCreateInfo computePipelineCreateInfo{};
  computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.layout = _pipelineLayout;
  computePipelineCreateInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
  //
  computePipelineCreateInfo.stage = shaderStage;
  if (vkCreateComputePipelines(_device->getLogicalDevice(), nullptr, 1, &computePipelineCreateInfo, nullptr,
                               &_pipeline) != VK_SUCCESS)
    throw std::runtime_error("failed to create compute pipeline!");
}