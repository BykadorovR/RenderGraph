module RenderPass;
import <volk.h>;
using namespace RenderGraph;

RenderPass::RenderPass(const Device& device) noexcept : _device(&device) {}

void RenderPass::_destroyFramebuffers() {
  for (auto& fb : _framebuffers) {
    vkDestroyFramebuffer(_device->getLogicalDevice(), fb, nullptr);
  }
  _framebuffers.clear();
}

void RenderPass::_createFramebuffers(const std::vector<std::vector<VkImageView>>& colorImageViews,
                                     const std::vector<VkImageView>& depthImageViews,
                                     VkExtent2D extent) {
  int count = (int)depthImageViews.size();
  for (auto& views : colorImageViews) count = std::max(count, (int)views.size());
  _framebuffers.resize(count);

  for (int i = 0; i < count; i++) {
    std::vector<VkImageView> attachmentViews;
    for (auto& colorViews : colorImageViews) {
      attachmentViews.push_back(colorViews[i % colorViews.size()]);
    }
    if (!depthImageViews.empty()) {
      attachmentViews.push_back(depthImageViews[i % depthImageViews.size()]);
    }

    VkFramebufferCreateInfo framebufferInfo{.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                            .renderPass = _renderPass,
                                            .attachmentCount = (uint32_t)attachmentViews.size(),
                                            .pAttachments = attachmentViews.data(),
                                            .width = extent.width,
                                            .height = extent.height,
                                            .layers = 1};
    if (vkCreateFramebuffer(_device->getLogicalDevice(), &framebufferInfo, nullptr, &_framebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

void RenderPass::create(const std::vector<VkFormat>& colorFormats,
                        const std::vector<bool>& clearColors,
                        const std::vector<VkImageLayout>& colorFinalLayouts,
                        std::optional<VkFormat> depthFormat,
                        bool clearDepth,
                        const std::vector<std::vector<VkImageView>>& colorImageViews,
                        const std::vector<VkImageView>& depthImageViews,
                        VkExtent2D extent) {
  std::vector<VkAttachmentDescription> attachments;
  for (int i = 0; i < (int)colorFormats.size(); i++) {
    attachments.push_back(VkAttachmentDescription{
        .format = colorFormats[i],
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = clearColors[i] ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        // Graph::render() always transitions attachments to GENERAL before execute(), so initialLayout = GENERAL.
        // finalLayout = GENERAL keeps the same layout semantics as dynamic rendering.
        .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
        .finalLayout = colorFinalLayouts[i]});
  }

  if (depthFormat) {
    attachments.push_back(
        VkAttachmentDescription{.format = depthFormat.value(),
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = clearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .finalLayout = VK_IMAGE_LAYOUT_GENERAL});
  }

  // Subpass references - use GENERAL layout (consistent with dynamic rendering approach)
  std::vector<VkAttachmentReference> colorRefs;
  for (int i = 0; i < (int)colorFormats.size(); i++) {
    colorRefs.push_back({(uint32_t)i, VK_IMAGE_LAYOUT_GENERAL});
  }

  VkSubpassDescription subpass{.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                               .colorAttachmentCount = (uint32_t)colorRefs.size(),
                               .pColorAttachments = colorRefs.data()};

  std::optional<VkAttachmentReference> depthRef;
  if (depthFormat) {
    depthRef = VkAttachmentReference{(uint32_t)colorFormats.size(), VK_IMAGE_LAYOUT_GENERAL};
    subpass.pDepthStencilAttachment = &depthRef.value();
  }

  VkSubpassDependency dependency{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};

  VkRenderPassCreateInfo renderPassInfo{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                        .attachmentCount = (uint32_t)attachments.size(),
                                        .pAttachments = attachments.data(),
                                        .subpassCount = 1,
                                        .pSubpasses = &subpass,
                                        .dependencyCount = 1,
                                        .pDependencies = &dependency};

  if (vkCreateRenderPass(_device->getLogicalDevice(), &renderPassInfo, nullptr, &_renderPass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }

  _createFramebuffers(colorImageViews, depthImageViews, extent);
}

void RenderPass::recreateFramebuffers(const std::vector<std::vector<VkImageView>>& colorImageViews,
                                      const std::vector<VkImageView>& depthImageViews,
                                      VkExtent2D extent) {
  _destroyFramebuffers();
  _createFramebuffers(colorImageViews, depthImageViews, extent);
}

VkRenderPass RenderPass::getRenderPass() const noexcept { return _renderPass; }

VkFramebuffer RenderPass::getFramebuffer(int index) const noexcept { return _framebuffers[index]; }

int RenderPass::getFramebufferCount() const noexcept { return (int)_framebuffers.size(); }

RenderPass::~RenderPass() {
  _destroyFramebuffers();
  if (_renderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(_device->getLogicalDevice(), _renderPass, nullptr);
  }
}
