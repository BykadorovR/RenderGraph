module Graph;
import <set>;
import <ranges>;
import <algorithm>;
using namespace RenderGraph;

void GraphStorage::add(std::string_view name, std::unique_ptr<ImageViewHolder> imageHolder) noexcept {
  _imageViewHolders[std::string(name)] = std::move(imageHolder);
}

void GraphStorage::add(std::string_view name, std::vector<std::unique_ptr<Buffer>>& buffers) noexcept {
  _buffers[std::string(name)] = std::move(buffers);
}

void GraphStorage::reset(std::vector<std::shared_ptr<ImageView>> oldSwapchain,
                         std::vector<std::shared_ptr<ImageView>> newSwapchain) noexcept {
  glm::ivec2 resolution = newSwapchain.front()->getImage().getResolution();
  auto nameSwapchain = find(oldSwapchain);
  if (nameSwapchain.empty() == false) {
    _imageViewHolders[nameSwapchain]->setImageViews(newSwapchain);
  }

  for (auto&& [name, value] : _imageViewHolders) {
    if (name != nameSwapchain) {
      auto indexFunction = value->getIndexFunction();
      auto imageViews = value->getImageViews();
      for (int i = 0; i < imageViews.size(); i++) {
        auto& image = imageViews[i]->getImage();
        if (image.getResolution() != resolution) {
          // recreate image
          image.destroy();
          image.createImage(image.getFormat(), resolution, image.getMipMapNumber(), image.getLayerNumber(),
                            image.getAspectMask(), image.getUsageFlags());
          imageViews[i]->destroy();
          imageViews[i]->createImageView(imageViews[i]->getType(), imageViews[i]->getBaseMipMap(),
                                         imageViews[i]->getBaseArrayLayer());
        }
      }
    }
  }
}

std::string GraphStorage::find(const std::vector<std::shared_ptr<ImageView>>& imageViews) noexcept {
  for (auto&& [name, imageViewHolder] : _imageViewHolders) {
    if (imageViewHolder->contains(imageViews)) return name;
  }
  return std::string{};
}

const ImageViewHolder& GraphStorage::getImageViewHolder(std::string_view name) const noexcept {
  return *_imageViewHolders.at(std::string(name));
}

std::vector<Buffer*> GraphStorage::getBuffer(std::string_view name) const noexcept {
  // std::ranges::to makes vector by itself
  return _buffers.at(std::string(name)) | std::views::transform([](auto& p) { return p.get(); }) |
         std::ranges::to<std::vector>();
}

GraphPass::GraphPass(std::string_view name, GraphPassType graphPassType, const GraphStorage& graphStorage) noexcept
    : _name(name),
      _graphPassType(graphPassType),
      _graphStorage(&graphStorage) {}

void GraphPass::registerGraphElement(std::shared_ptr<GraphElement> graphElement) noexcept {
  _graphElements.push_back(graphElement);
}

std::string GraphPass::getName() const noexcept { return _name; }

void GraphPass::reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) {
  for (auto&& graphElement : _graphElements) {
    graphElement->reset(swapchain);
  }
}

void GraphPass::addSignalSemaphore(std::vector<std::shared_ptr<Semaphore>>& signalSemaphore,
                                   std::function<int()> index) noexcept {
  _signalSemaphores.push_back({signalSemaphore, index});
}

void GraphPass::addWaitSemaphore(std::vector<std::shared_ptr<Semaphore>>& waitSemaphore,
                                 std::function<int()> index) noexcept {
  _waitSemaphores.push_back({waitSemaphore, index});
}

GraphPassType GraphPass::getGraphPassType() const noexcept { return _graphPassType; }

std::vector<Semaphore*> GraphPass::getSignalSemaphores() const noexcept {
  // similar to getBuffer
  return _signalSemaphores | std::views::transform([](auto& pair) {
           auto& [semaphores, index] = pair;
           return semaphores[index()].get();
         }) |
         std::ranges::to<std::vector>();
}

std::vector<Semaphore*> GraphPass::getWaitSemaphores() const noexcept {
  return _waitSemaphores | std::views::transform([](auto& pair) {
           auto& [semaphores, index] = pair;
           return semaphores[index()].get();
         }) |
         std::ranges::to<std::vector>();
}

std::vector<CommandBuffer*> GraphPass::getCommandBuffers() const noexcept {
  return _commandBuffers | std::views::transform([](auto& p) { return p.get(); }) | std::ranges::to<std::vector>();
}

GraphPassGraphic::GraphPassGraphic(std::string_view name,
                                   int maxFramesInFlight,
                                   const GraphStorage& graphStorage,
                                   const Device& device) noexcept
    : GraphPass(name, GraphPassType::GRAPHIC, graphStorage) {
  _device = &device;
  _commandPool = std::make_unique<CommandPool>(vkb::QueueType::graphics, device);
  _commandBuffers.resize(maxFramesInFlight);
  std::ranges::generate(_commandBuffers, [&] { return std::make_unique<CommandBuffer>(*_commandPool, device); });
  _pipelineGraphic = std::make_unique<PipelineGraphic>();
}

void GraphPassGraphic::addColorTarget(std::string_view name) noexcept { _colorTargets.emplace_back(name); }

void GraphPassGraphic::setDepthTarget(std::string_view name) noexcept { _depthTarget = name; }

void GraphPassGraphic::addTextureInput(std::string_view name) noexcept { _textureInputs.emplace_back(name); }

void GraphPassGraphic::clearTarget(std::string_view name) noexcept { _clearTarget[std::string(name)] = true; }

const std::vector<std::string>& GraphPassGraphic::getColorTargets() const noexcept { return _colorTargets; }

std::optional<std::string> GraphPassGraphic::getDepthTarget() const noexcept { return _depthTarget; }

const std::vector<std::string>& GraphPassGraphic::getTextureInputs() const noexcept { return _textureInputs; }

PipelineGraphic& GraphPassGraphic::getPipelineGraphic(const GraphStorage& graphStorage) const noexcept {
  auto colorFormats = _colorTargets | std::views::transform([&](auto& colorTarget) {
                        return graphStorage.getImageViewHolder(colorTarget).getImageView().getImage().getFormat();
                      }) |
                      std::ranges::to<std::vector>();
  _pipelineGraphic->setColorAttachments(colorFormats);

  std::optional<VkFormat> depthFormat = std::nullopt;
  if (_depthTarget) {
    depthFormat = graphStorage.getImageViewHolder(_depthTarget.value()).getImageView().getImage().getFormat();
  }
  _pipelineGraphic->setDepthAttachment(depthFormat);

  return *_pipelineGraphic;
}

const std::unordered_map<std::string, bool>& GraphPassGraphic::getClearTargets() const noexcept {
  return _clearTarget;
}

void GraphPassGraphic::setRenderPass(RenderPass* renderPass) noexcept { _renderPass = renderPass; }

void GraphPassGraphic::execute(int currentFrame, const CommandBuffer& commandBuffer) {
  for (auto&& graphElement : _graphElements) graphElement->update(currentFrame, commandBuffer);

  auto resolution = _graphStorage->getImageViewHolder(_colorTargets.front()).getImageView().getImage().getResolution();

  if (_renderPass != nullptr) {
    int fbIndex = _graphStorage->getImageViewHolder(_colorTargets.front()).getIndex();

    std::vector<VkClearValue> clearValues;
    bool hasClear = false;
    for (auto& colorTarget : _colorTargets) {
      VkClearValue cv{};
      if (_clearTarget.contains(colorTarget) && _clearTarget.at(colorTarget)) { cv.color = {0.f, 0.f, 0.f, 1.f}; hasClear = true; }
      clearValues.push_back(cv);
    }
    if (_depthTarget) {
      VkClearValue cv{};
      if (_clearTarget.contains(_depthTarget.value()) && _clearTarget.at(_depthTarget.value())) { cv.depthStencil = {1.f, 0}; hasClear = true; }
      clearValues.push_back(cv);
    }

    VkRenderPassBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = _renderPass->getRenderPass(),
        .framebuffer = _renderPass->getFramebuffer(fbIndex),
        .renderArea = {.offset = {0, 0}, .extent = {(uint32_t)resolution.x, (uint32_t)resolution.y}},
        .clearValueCount = hasClear ? (uint32_t)clearValues.size() : 0,
        .pClearValues = hasClear ? clearValues.data() : nullptr};
    vkCmdBeginRenderPass(commandBuffer.getCommandBuffer(), &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
    for (auto&& graphElement : _graphElements) graphElement->draw(currentFrame, commandBuffer);
    vkCmdEndRenderPass(commandBuffer.getCommandBuffer());
  } else {
    auto createColorAttachment = [this](const auto& colorTarget) {
      auto& imageViewHolder = _graphStorage->getImageViewHolder(colorTarget);
      VkRenderingAttachmentInfo info{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                     .imageView = imageViewHolder.getImageView().getImageView(),
                                     .imageLayout = imageViewHolder.getImageView().getImage().getImageLayout(),
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
      if (_clearTarget.contains(colorTarget) && _clearTarget.at(colorTarget)) {
        info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        info.clearValue.color = {0.f, 0.f, 0.f, 1.f};
      }
      return info;
    };

    auto colorAttachments = _colorTargets | std::views::transform(createColorAttachment) |
                            std::ranges::to<std::vector>();

    std::optional<VkRenderingAttachmentInfo> depthAttachment;
    if (_depthTarget) {
      auto& imageViewHolder = _graphStorage->getImageViewHolder(_depthTarget.value());
      depthAttachment = VkRenderingAttachmentInfo{
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = imageViewHolder.getImageView().getImageView(),
          .imageLayout = imageViewHolder.getImageView().getImage().getImageLayout(),
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
      if (_clearTarget.contains(_depthTarget.value()) && _clearTarget.at(_depthTarget.value())) {
        depthAttachment->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment->clearValue.depthStencil = {1.f, 0};
      }
    }

    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = {(uint32_t)resolution.x, (uint32_t)resolution.y}},
        .layerCount = 1,
        .colorAttachmentCount = (uint32_t)colorAttachments.size(),
        .pColorAttachments = colorAttachments.data()};
    if (depthAttachment) renderingInfo.pDepthAttachment = &depthAttachment.value();

    vkCmdBeginRendering(commandBuffer.getCommandBuffer(), &renderingInfo);
    for (auto&& graphElement : _graphElements) graphElement->draw(currentFrame, commandBuffer);
    vkCmdEndRendering(commandBuffer.getCommandBuffer());
  }
}

GraphPassCompute::GraphPassCompute(std::string_view name,
                                   int maxFramesInFlight,
                                   bool separate,
                                   const GraphStorage& graphStorage,
                                   const Device& device) noexcept
    : GraphPass(name, GraphPassType::COMPUTE, graphStorage) {
  _device = &device;
  _separate = separate;

  auto queueType = vkb::QueueType::graphics;
  if (separate) queueType = vkb::QueueType::compute;
  _commandPool = std::make_unique<CommandPool>(queueType, device);
  _commandBuffers.resize(maxFramesInFlight);
  std::ranges::generate(_commandBuffers, [&] { return std::make_unique<CommandBuffer>(*_commandPool, device); });
}

void GraphPassCompute::addStorageBufferInput(std::string_view name) noexcept {
  _storageBufferInputs.emplace_back(name);
}

void GraphPassCompute::addStorageTextureInput(std::string_view name) noexcept {
  _storageTextureInputs.emplace_back(name);
}

void GraphPassCompute::addStorageBufferOutput(std::string_view name) noexcept {
  _storageBufferOutputs.emplace_back(name);
}

void GraphPassCompute::addStorageTextureOutput(std::string_view name) noexcept {
  _storageTextureOutputs.emplace_back(name);
}

const std::vector<std::string>& GraphPassCompute::getStorageBufferInputs() const noexcept {
  return _storageBufferInputs;
}

const std::vector<std::string>& GraphPassCompute::getStorageBufferOutputs() const noexcept {
  return _storageBufferOutputs;
}

const std::vector<std::string>& GraphPassCompute::getStorageTextureInputs() const noexcept {
  return _storageTextureInputs;
}

const std::vector<std::string>& GraphPassCompute::getStorageTextureOutputs() const noexcept {
  return _storageTextureOutputs;
}

bool GraphPassCompute::isSeparate() const noexcept { return _separate; }

void GraphPassCompute::execute(int currentFrame, const CommandBuffer& commandBuffer) {
  for (auto&& graphElement : _graphElements) {
    graphElement->update(currentFrame, commandBuffer);
    graphElement->draw(currentFrame, commandBuffer);
  }
}

Graph::Graph(int threadsNumber,
             int maxFramesInFlight,
             Swapchain& swapchain,
             const Window& window,
             const Device& device) noexcept
    : _swapchain(&swapchain),
      _window(&window),
      _device(&device) {
  _threadPool = std::make_unique<BS::thread_pool>(threadsNumber);
  _timestamps = std::make_unique<Timestamps>(device);
  _graphStorage = std::make_unique<GraphStorage>();
  _maxFramesInFlight = maxFramesInFlight;
}

void Graph::initialize() noexcept {
  // create 3 special semaphores
  std::ranges::generate_n(std::back_inserter(_semaphoreImageAvailable), _maxFramesInFlight,
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
  std::ranges::generate_n(std::back_inserter(_semaphoreRenderFinished), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
  _semaphoreInFlight = std::make_unique<Semaphore>(VK_SEMAPHORE_TYPE_TIMELINE, *_device);
}

void Graph::setRenderingMode(RenderingMode mode) noexcept { _renderingMode = mode; }

RenderingMode Graph::getRenderingMode() const noexcept { return _renderingMode; }

GraphStorage& Graph::getGraphStorage() const noexcept { return *_graphStorage; }

std::unordered_map<std::string, glm::dvec2> Graph::getTimestamps() const noexcept {
  return _timestamps->getTimestamps();
}

int Graph::getFrameInFlight() const noexcept { return _frameInFlight; }

GraphPassGraphic& Graph::createPassGraphic(std::string_view name) {
  auto it = std::find_if(_passes.begin(), _passes.end(),
                         [name = name](std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == name; });
  if (it != _passes.end()) return static_cast<GraphPassGraphic&>(**it);

  _passes.push_back(std::make_unique<GraphPassGraphic>(name, _maxFramesInFlight, *_graphStorage, *_device));
  return static_cast<GraphPassGraphic&>(*_passes.back());
}

GraphPassCompute& Graph::createPassCompute(std::string_view name, bool separate) {
  auto it = std::find_if(_passes.begin(), _passes.end(),
                         [name = name](std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == name; });
  if (it != _passes.end()) return static_cast<GraphPassCompute&>(**it);

  _passes.push_back(std::make_unique<GraphPassCompute>(name, _maxFramesInFlight, separate, *_graphStorage, *_device));
  return static_cast<GraphPassCompute&>(*_passes.back());
}

GraphPassGraphic* Graph::getPassGraphic(std::string_view name) const noexcept {
  auto it = std::find_if(_passes.begin(), _passes.end(),
                         [name](const std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == name; });
  if (it != _passes.end()) {
    return static_cast<GraphPassGraphic*>((*it).get());
  }

  return nullptr;
}

GraphPassCompute* Graph::getPassCompute(std::string_view name) const noexcept {
  auto it = std::find_if(_passes.begin(), _passes.end(),
                         [name](const std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == name; });
  if (it != _passes.end()) {
    return static_cast<GraphPassCompute*>((*it).get());
  }

  return nullptr;
}

void Graph::print() const noexcept {
  if (_passesOrdered.empty()) return;

  auto printImages = [this](const auto& keys, std::string_view keyTag) {
    for (auto&& name : keys) {
      std::cout << keyTag << name << "; ";
      for (auto&& imageView : _graphStorage->getImageViewHolder(name).getImageViews()) {
        std::cout << imageView->getImage().getImage() << " ";
      }
      std::cout << std::endl;
    }
  };
  auto printBuffers = [this](const auto& keys, std::string_view keyTag) {
    for (auto&& name : keys) {
      std::cout << keyTag << name << "; ";
      for (auto&& resource : _graphStorage->getBuffer(name)) {
        std::cout << resource->getBuffer() << " ";
      }
      std::cout << std::endl;
    }
  };
  auto printRange = [](std::string_view label, const auto& range, auto getter) {
    std::cout << label;
    for (auto&& item : range) {
      std::cout << getter(item) << " ";
    }
    std::cout << std::endl;
  };

  for (auto&& value : _passesOrdered) {
    std::cout << "Name: " << value->getName()
              << ", Stage : " << (value->getGraphPassType() == GraphPassType::GRAPHIC ? "GRAPHIC" : "COMPUTE")
              << std::endl;
    if (value->getGraphPassType() == GraphPassType::COMPUTE)
      std::cout << " separate: " << (static_cast<GraphPassCompute*>(value)->isSeparate() ? "true" : "false")
                << std::endl;

    printRange(" wait semaphores: ", value->getWaitSemaphores(), [](auto* s) { return s->getSemaphore(); });
    printRange(" signal semaphores: ", value->getSignalSemaphores(), [](auto* s) { return s->getSemaphore(); });
    printRange(" command buffers: ", value->getCommandBuffers(), [](auto* c) { return c->getCommandBuffer(); });
    if (value->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto passGraphic = static_cast<GraphPassGraphic*>(value);
      printImages(passGraphic->getColorTargets(), " color target: ");
      {
        auto name = passGraphic->getDepthTarget();
        if (name)
          std::cout << " depth target: " << name.value() << "; "
                    << _graphStorage->getImageViewHolder(name.value()).getImageView().getImage().getImage()
                    << std::endl;
      }
      printImages(passGraphic->getTextureInputs(), " texture input: ");
    }

    if (value->getGraphPassType() == GraphPassType::COMPUTE) {
      auto passCompute = static_cast<GraphPassCompute*>(value);
      printBuffers(passCompute->getStorageBufferInputs(), " storage buffer input: ");
      printBuffers(passCompute->getStorageBufferOutputs(), " storage buffer output: ");
      printImages(passCompute->getStorageTextureInputs(), " storage texture input: ");
      printImages(passCompute->getStorageTextureOutputs(), " storage texture output: ");
    }
  }
}

void Graph::calculate() {
  auto getInputs = [this](std::string_view nameNode) -> std::vector<std::string> {
    auto it = std::find_if(
        _passes.rbegin(), _passes.rend(),
        [nameNode = nameNode](std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == nameNode; });
    auto value = (*it).get();

    std::vector<std::string> dependencies;
    if (value->getGraphPassType() == GraphPassType::COMPUTE) {
      auto passCompute = static_cast<GraphPassCompute*>(value);
      for (auto&& nameResource : passCompute->getStorageBufferInputs()) {
        dependencies.push_back(nameResource);
      }
      for (auto&& nameResource : passCompute->getStorageTextureInputs()) {
        dependencies.push_back(nameResource);
      }
    }

    if (value->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto passGraphic = static_cast<GraphPassGraphic*>(value);
      for (auto&& nameResource : passGraphic->getTextureInputs()) {
        dependencies.push_back(nameResource);
      }
    }

    return dependencies;
  };

  auto findTarget = [](std::vector<GraphPass*> passes, std::string_view findName) -> GraphPass* {
    for (auto pass : std::views::reverse(passes)) {
      if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
        auto passGraphic = static_cast<GraphPassGraphic*>(pass);
        if (std::ranges::contains(passGraphic->getColorTargets(), findName) ||
            passGraphic->getDepthTarget() == findName)
          return pass;
      }

      if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
        auto passCompute = static_cast<GraphPassCompute*>(pass);
        if (std::ranges::contains(passCompute->getStorageBufferOutputs(), findName) ||
            std::ranges::contains(passCompute->getStorageTextureOutputs(), findName))
          return pass;
      }
    }
    return nullptr;
  };

  GraphPass* root = _passes.back().get();
  auto passesBackup = _passes | std::views::transform([](auto& p) { return p.get(); }) |
                      std::ranges::to<std::vector<GraphPass*>>();
  // we should for every root pass run traversal process
  std::function<void(GraphPass*)> traverse = [&](GraphPass* node) {
    if (passesBackup.empty()) return;

    passesBackup.erase(std::remove(passesBackup.begin(), passesBackup.end(), node), passesBackup.end());
    _passesOrdered.push_front(node);

    auto inputs = getInputs(node->getName());
    if (inputs.empty()) {
      // this means that stage depends only on it's frame buffer attachment
      if (node->getGraphPassType() == GraphPassType::GRAPHIC) {
        auto passGraphic = static_cast<GraphPassGraphic*>(node);
        inputs = passGraphic->getColorTargets();
      }
    }
    std::vector<GraphPass*> passNext =
        inputs | std::views::transform([&](auto& input) { return findTarget(passesBackup, input); }) |
        std::views::filter([](GraphPass* p) { return p != nullptr; }) | std::ranges::to<std::vector>();

    for (auto&& pass : passNext) traverse(pass);
  };

  if (root) traverse(root);

  // initialize render passes for RENDER_PASS mode
  if (_renderingMode == RenderingMode::RENDER_PASS) {
    const auto swapchainImageViews = _swapchain->getImageViews();
    for (auto&& pass : _passesOrdered) {
      if (pass->getGraphPassType() != GraphPassType::GRAPHIC) continue;
      auto* passGraphic = static_cast<GraphPassGraphic*>(pass);

      std::vector<VkFormat> colorFormats;
      std::vector<bool> clearColors;
      std::vector<VkImageLayout> colorFinalLayouts;
      std::vector<std::vector<VkImageView>> colorImageViews;
      for (auto& colorTarget : passGraphic->getColorTargets()) {
        auto& holder = _graphStorage->getImageViewHolder(colorTarget);
        colorFormats.push_back(holder.getImageView().getImage().getFormat());
        clearColors.push_back(passGraphic->getClearTargets().contains(colorTarget) && passGraphic->getClearTargets().at(colorTarget));
        colorFinalLayouts.push_back(VK_IMAGE_LAYOUT_GENERAL);
        auto views = holder.getImageViews();
        std::vector<VkImageView> vkViews;
        vkViews.reserve(views.size());
        for (auto* iv : views) vkViews.push_back(iv->getImageView());
        colorImageViews.push_back(std::move(vkViews));
      }

      std::optional<VkFormat> depthFormat;
      std::vector<VkImageView> depthImageViews;
      bool clearDepth = false;
      if (auto depthTarget = passGraphic->getDepthTarget()) {
        auto& holder = _graphStorage->getImageViewHolder(depthTarget.value());
        depthFormat = holder.getImageView().getImage().getFormat();
        clearDepth = passGraphic->getClearTargets().contains(depthTarget.value()) && passGraphic->getClearTargets().at(depthTarget.value());
        auto views = holder.getImageViews();
        depthImageViews.reserve(views.size());
        for (auto* iv : views) depthImageViews.push_back(iv->getImageView());
      }

      auto resolution = _graphStorage->getImageViewHolder(passGraphic->getColorTargets().front())
                            .getImageView()
                            .getImage()
                            .getResolution();
      VkExtent2D extent{(uint32_t)resolution.x, (uint32_t)resolution.y};

      auto renderPass = std::make_unique<RenderPass>(*_device);
      renderPass->create(colorFormats, clearColors, colorFinalLayouts, depthFormat, clearDepth, colorImageViews,
                         depthImageViews, extent);
      passGraphic->getPipelineGraphic(*_graphStorage).setRenderPass(renderPass->getRenderPass());
      passGraphic->setRenderPass(renderPass.get());
      _renderPasses[passGraphic->getName()] = std::move(renderPass);
    }
  }

  // set semaphores between passes
  bool flagWaitForSwapchain = true;
  bool queueTypeChange = false;
  GraphPass* previousPass = nullptr;
  for (auto&& pass : _passesOrdered) {
    // find if we need to change queue -> add semaphore
    if (previousPass) {
      if (pass->getGraphPassType() != previousPass->getGraphPassType()) {
        if ((previousPass->getGraphPassType() == GraphPassType::COMPUTE &&
             static_cast<GraphPassCompute*>(previousPass)->isSeparate()) ||
            (pass->getGraphPassType() == GraphPassType::COMPUTE && static_cast<GraphPassCompute*>(pass)->isSeparate()))
          queueTypeChange = true;
      } else {
        if (previousPass->getGraphPassType() == GraphPassType::COMPUTE &&
            pass->getGraphPassType() == GraphPassType::COMPUTE &&
            static_cast<GraphPassCompute*>(previousPass)->isSeparate() !=
                static_cast<GraphPassCompute*>(pass)->isSeparate())
          queueTypeChange = true;
      }
    }

    _cache[pass] = {queueTypeChange, previousPass};

    // signal semaphore for the previous pass
    // wait semaphore for the current pass
    // only if there are separate queues for compute and graphic
    if (queueTypeChange) {
      std::vector<std::shared_ptr<Semaphore>> semaphoreQueueType(_maxFramesInFlight);
      std::ranges::generate(semaphoreQueueType,
                            [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
      pass->addWaitSemaphore(semaphoreQueueType, [this]() { return _frameInFlight; });
      previousPass->addSignalSemaphore(semaphoreQueueType, [this]() { return _frameInFlight; });
      queueTypeChange = false;
    }
    // special case if we read from swapchain
    // who first interact with swapchain that should wait for the semaphore
    if (flagWaitForSwapchain) {
      bool swapchainFound = false;
      auto checkSwapchain = [this](const auto& input) -> bool {
        for (auto&& name : input)
          if (_graphStorage->getImageViewHolder(name).contains(_swapchain->getImageViews())) return true;
        return false;
      };
      if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
        auto passGraphic = static_cast<GraphPassGraphic*>(pass);
        if (checkSwapchain(passGraphic->getColorTargets()) || checkSwapchain(passGraphic->getTextureInputs()))
          swapchainFound = true;
      }
      if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
        auto passCompute = static_cast<GraphPassCompute*>(pass);
        if (checkSwapchain(passCompute->getStorageTextureInputs()) ||
            checkSwapchain(passCompute->getStorageTextureOutputs()))
          swapchainFound = true;
      }
      if (swapchainFound) {
        pass->addWaitSemaphore(_semaphoreImageAvailable, [this]() { return _frameInFlight; });
        flagWaitForSwapchain = false;
      }
    }
    // end node should signal end semaphore
    if (pass == root) {
      pass->addSignalSemaphore(_semaphoreRenderFinished, [this]() { return _swapchain->getSwapchainIndex(); });
    }

    previousPass = pass;
  }
}

bool Graph::render() {
  // timeline semaphore instead of fence
  if (_valueSemaphoreInFlight > _maxFramesInFlight) {
    uint64_t waitValue = _valueSemaphoreInFlight - _maxFramesInFlight;
    auto semaphoreInFlight = _semaphoreInFlight->getSemaphore();
    VkSemaphoreWaitInfo waitInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &semaphoreInFlight,
        .pValues = &waitValue,
    };

    vkWaitSemaphores(_device->getLogicalDevice(), &waitInfo, std::numeric_limits<std::uint64_t>::max());
  }

  auto status = _swapchain->acquireNextImage(*_semaphoreImageAvailable[_frameInFlight]);
  // notify about reset needed
  if (status == VK_ERROR_OUT_OF_DATE_KHR) {
    return true;
  } else if (status != VK_SUCCESS && status != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image");
  }

  auto swapchainIndex = _swapchain->getSwapchainIndex();
  _timestamps->resetQueryPool();
  // run all passes' execution functions
  std::vector<std::future<void>> futureTasks =
      _passesOrdered | std::views::transform([this, swapchainIndex](auto& pass) {
        auto commandBuffer = pass->getCommandBuffers()[_frameInFlight];
        if (commandBuffer->getActive() == false) commandBuffer->beginCommands();

        // change layouts if needed (attachments and/or swapchain), handles reset as well
        if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
          auto passGraphic = static_cast<GraphPassGraphic*>(pass);
          auto colorTargets = passGraphic->getColorTargets();
          auto transitionToGeneral = [&](auto& iv, VkAccessFlags dstMask) {
            if (iv.getImage().getImageLayout() != VK_IMAGE_LAYOUT_GENERAL)
              iv.getImage().changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_NONE,
                                         dstMask, *commandBuffer);
          };
          auto colorDst = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
          auto depthDst = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
          auto depthTarget = passGraphic->getDepthTarget();

          if (_renderingMode == RenderingMode::RENDER_PASS) {
            // Transition only the view used by this frame's framebuffer (same fbIndex as execute())
            int fbIndex = _graphStorage->getImageViewHolder(colorTargets.front()).getIndex();
            auto transitionAtIndex = [&](const auto& holder, VkAccessFlags dstMask) {
              auto views = holder.getImageViews();
              transitionToGeneral(*views[fbIndex % views.size()], dstMask);
            };
            for (auto&& colorTarget : colorTargets)
              transitionAtIndex(_graphStorage->getImageViewHolder(colorTarget), colorDst);
            if (depthTarget)
              transitionAtIndex(_graphStorage->getImageViewHolder(depthTarget.value()), depthDst);
          } else {
            for (auto&& colorTarget : colorTargets)
              transitionToGeneral(_graphStorage->getImageViewHolder(colorTarget).getImageView(), colorDst);
            if (depthTarget)
              transitionToGeneral(_graphStorage->getImageViewHolder(depthTarget.value()).getImageView(), depthDst);
          }
        }

        return _threadPool->submit([this, pass, commandBuffer]() {
          _timestamps->pushTimestamp(pass->getName(), *commandBuffer);
          pass->execute(_frameInFlight, *commandBuffer);
          _timestamps->popTimestamp(pass->getName(), *commandBuffer);
        });
      }) |
      std::ranges::to<std::vector<std::future<void>>>();

  auto submitPassToQueue = [this](GraphPass* previousPass, const std::vector<CommandBuffer*>& commandBufferSubmit,
                                  const std::vector<VkSemaphore>& waitSemaphores,
                                  const std::vector<VkSemaphore>& signalSemaphores,
                                  std::optional<VkTimelineSemaphoreSubmitInfo> signalTimelineInfo = std::nullopt) {
    // need to end command buffers before submit
    std::vector<VkCommandBuffer> commandBufferRawSubmit;
    commandBufferRawSubmit.reserve(commandBufferSubmit.size());
    for (auto&& commandBuffer : commandBufferSubmit) {
      commandBuffer->endCommands();
      commandBufferRawSubmit.push_back(commandBuffer->getCommandBuffer());
    }

    // determine wait stages
    std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    auto queueType = vkb::QueueType::graphics;
    if (previousPass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto passComputePrevious = static_cast<GraphPassCompute*>(previousPass);
      std::ranges::fill(waitStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      if (passComputePrevious->isSeparate()) queueType = vkb::QueueType::compute;
    }

    // submit + semaphores
    VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .pNext = signalTimelineInfo ? &signalTimelineInfo.value() : nullptr,
                            .waitSemaphoreCount = (uint32_t)waitSemaphores.size(),
                            .pWaitSemaphores = waitSemaphores.data(),
                            .pWaitDstStageMask = waitStages.data(),
                            .commandBufferCount = (uint32_t)commandBufferRawSubmit.size(),
                            .pCommandBuffers = commandBufferRawSubmit.data(),
                            .signalSemaphoreCount = (uint32_t)signalSemaphores.size(),
                            .pSignalSemaphores = signalSemaphores.data()};

    vkQueueSubmit(_device->getQueue(queueType), 1, &submitInfo, nullptr);
  };

  // command buffer from passes
  std::vector<CommandBuffer*> commandBufferSubmit;
  std::vector<VkSemaphore> signalSemaphores;
  std::vector<VkSemaphore> waitSemaphores;
  // submit recorded command buffer to GPU
  for (auto&& [pass, futureTask] : std::views::zip(_passesOrdered, futureTasks)) {
    // wait execution of current render pass
    if (futureTask.valid()) futureTask.get();

    auto [queueTypeChange, previousPass] = _cache[pass];

    if (previousPass) {
      if (queueTypeChange) {
        submitPassToQueue(previousPass, commandBufferSubmit, waitSemaphores, signalSemaphores);
        //
        commandBufferSubmit.clear();
        signalSemaphores.clear();
        waitSemaphores.clear();
      } else {
        // put EXECUTION AND MEMORY barriers if needed (not layout transition ones)
        // IMPORTANT: we should add any barrier to the previous stage because potentially all command buffer are already
        // recorded. So we need to add barrier to the end of the previous command buffer.
        auto calculateImageBarriers = [this](auto range, VkAccessFlags srcMask, VkAccessFlags dstMask) {
          std::vector<VkImageMemoryBarrier> imageBarriers;
          for (auto&& item : range) {
            auto& image = _graphStorage->getImageViewHolder(item).getImageView().getImage();
            imageBarriers.push_back(VkImageMemoryBarrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                         .srcAccessMask = srcMask,
                                                         .dstAccessMask = dstMask,
                                                         .oldLayout = image.getImageLayout(),
                                                         .newLayout = image.getImageLayout(),
                                                         .image = image.getImage(),
                                                         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
          }

          return imageBarriers;
        };

        auto calculateBufferBarriers = [this](auto range, VkAccessFlags srcMask, VkAccessFlags dstMask) {
          std::vector<VkBufferMemoryBarrier> bufferBarriers;
          for (auto&& item : range) {
            auto buffer = _graphStorage->getBuffer(item)[_frameInFlight];
            bufferBarriers.push_back(VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                                           .srcAccessMask = srcMask,
                                                           .dstAccessMask = dstMask,
                                                           .buffer = buffer->getBuffer(),
                                                           .size = buffer->getSize()});
          }

          return bufferBarriers;
        };

        if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
          auto graphPassGraphic = static_cast<GraphPassGraphic*>(pass);
          auto imageBarriers = calculateImageBarriers(graphPassGraphic->getTextureInputs(),
                                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
          vkCmdPipelineBarrier(commandBufferSubmit.back()->getCommandBuffer(),
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                               0, nullptr, 0, nullptr, imageBarriers.size(), imageBarriers.data());
        }

        if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
          auto graphPassCompute = static_cast<GraphPassCompute*>(pass);
          auto imageBarriers = calculateImageBarriers(graphPassCompute->getStorageTextureInputs(),
                                                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
          auto bufferBarriers = calculateBufferBarriers(graphPassCompute->getStorageBufferInputs(),
                                                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

          vkCmdPipelineBarrier(previousPass->getCommandBuffers()[_frameInFlight]->getCommandBuffer(),
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                               nullptr, bufferBarriers.size(), bufferBarriers.data(), imageBarriers.size(),
                               imageBarriers.data());
        }
      }
    }

    commandBufferSubmit.push_back(pass->getCommandBuffers()[_frameInFlight]);
    for (auto&& semaphore : pass->getSignalSemaphores()) signalSemaphores.push_back(semaphore->getSemaphore());
    for (auto&& semaphore : pass->getWaitSemaphores()) waitSemaphores.push_back(semaphore->getSemaphore());
  }

  // last pass changes swapchain layout if needed
  if (_swapchain->getImage(swapchainIndex).getImageLayout() != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
    _swapchain->getImage(swapchainIndex)
        .changeLayout(_swapchain->getImage(swapchainIndex).getImageLayout(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_NONE, *commandBufferSubmit.back());
  }

  // submit last pass
  std::vector<uint64_t> signalValues(signalSemaphores.size() + 1, 0);
  signalValues[signalSemaphores.size()] = _valueSemaphoreInFlight;
  signalSemaphores.push_back(_semaphoreInFlight->getSemaphore());
  VkTimelineSemaphoreSubmitInfo timelineSignalInfo = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                                      .signalSemaphoreValueCount = (uint32_t)signalSemaphores.size(),
                                                      .pSignalSemaphoreValues = signalValues.data()};

  submitPassToQueue(_passesOrdered.back(), commandBufferSubmit, waitSemaphores, signalSemaphores, timelineSignalInfo);
  _timestamps->fetchTimestamps();

  auto semaphoreRenderFinished = _semaphoreRenderFinished[swapchainIndex]->getSemaphore();
  VkSwapchainKHR swapChains[] = {_swapchain->getSwapchain()};
  VkPresentInfoKHR presentInfo{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                               .waitSemaphoreCount = 1,
                               .pWaitSemaphores = &semaphoreRenderFinished,
                               .swapchainCount = 1,
                               .pSwapchains = swapChains,
                               .pImageIndices = &swapchainIndex};

  _valueSemaphoreInFlight++;
  _frameInFlight = (_valueSemaphoreInFlight - 1) % _maxFramesInFlight;

  auto result = vkQueuePresentKHR(_device->getQueue(vkb::QueueType::present), &presentInfo);
  if (result != VK_SUCCESS) {
    return true;
  }

  return false;
}

void Graph::reset() {
  // wait all queues idle before reset
  if (vkDeviceWaitIdle(_device->getLogicalDevice()) != VK_SUCCESS) throw std::runtime_error("failed to reset");

  if (_window->getResolution().x == 0 || _window->getResolution().y == 0)
    throw std::runtime_error("Can't reset if resolution is 0");

  auto oldSwapchain = _swapchain->reset(_window->getResolution());
  _graphStorage->reset(oldSwapchain, _swapchain->getImageViews());
  for (auto&& pass : _passesOrdered) {
    pass->reset(_swapchain->getImageViews());
  }

  for (auto& [passName, renderPass] : _renderPasses) {
    auto* passGraphic = getPassGraphic(passName);

    std::vector<std::vector<VkImageView>> colorImageViews;
    for (auto& colorTarget : passGraphic->getColorTargets()) {
      auto views = _graphStorage->getImageViewHolder(colorTarget).getImageViews();
      std::vector<VkImageView> vkViews;
      vkViews.reserve(views.size());
      for (auto* iv : views) vkViews.push_back(iv->getImageView());
      colorImageViews.push_back(std::move(vkViews));
    }

    std::vector<VkImageView> depthImageViews;
    if (auto depthTarget = passGraphic->getDepthTarget()) {
      auto views = _graphStorage->getImageViewHolder(depthTarget.value()).getImageViews();
      depthImageViews.reserve(views.size());
      for (auto* iv : views) depthImageViews.push_back(iv->getImageView());
    }

    auto resolution = _graphStorage->getImageViewHolder(passGraphic->getColorTargets().front())
                          .getImageView()
                          .getImage()
                          .getResolution();
    VkExtent2D extent{(uint32_t)resolution.x, (uint32_t)resolution.y};
    renderPass->recreateFramebuffers(colorImageViews, depthImageViews, extent);
  }
}