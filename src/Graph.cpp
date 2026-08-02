module Graph;
import <set>;
import <ranges>;
import <limits>;
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
          imageViews[i]->destroy();
          image.destroy();
          image.createImage(image.getFormat(), resolution, image.getMipMapNumber(), image.getLayerNumber(),
                            image.getAspectMask(), image.getUsageFlags());
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

bool GraphStorage::containsImageViewHolder(std::string_view name) const noexcept {
  return _imageViewHolders.contains(std::string(name));
}

const ImageViewHolder& GraphStorage::getImageViewHolder(std::string_view name) const {
  return *_imageViewHolders.at(std::string(name));
}

bool GraphStorage::containsBuffer(std::string_view name) const noexcept { return _buffers.contains(std::string(name)); }

std::vector<Buffer*> GraphStorage::getBuffer(std::string_view name) const {
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

GraphPassType GraphPass::getGraphPassType() const noexcept { return _graphPassType; }

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

void GraphPassGraphic::execute(int currentFrame, const CommandBuffer& commandBuffer) {
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

  std::vector<VkRenderingAttachmentInfo> colorAttachments = _colorTargets |
                                                            std::views::transform(createColorAttachment) |
                                                            std::ranges::to<std::vector>();

  std::optional<VkRenderingAttachmentInfo> depthAttachment = std::nullopt;
  if (_depthTarget.has_value()) {
    auto& target = _depthTarget.value();
    auto& imageViewHolder = _graphStorage->getImageViewHolder(target);

    depthAttachment = VkRenderingAttachmentInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = imageViewHolder.getImageView().getImageView(),
        .imageLayout = imageViewHolder.getImageView().getImage().getImageLayout(),
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    if (_clearTarget.contains(target) && _clearTarget.at(target)) {
      depthAttachment->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      depthAttachment->clearValue.depthStencil = {1.f, 0};
    }
  }

  VkRenderingInfo renderingInfo = {};
  renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  renderingInfo.renderArea.offset = {0, 0};
  // take image resolution, it's safe
  auto resolution = _graphStorage->getImageViewHolder(_colorTargets.front()).getImageView().getImage().getResolution();
  renderingInfo.renderArea.extent = VkExtent2D(resolution.x, resolution.y);
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = colorAttachments.size();
  renderingInfo.pColorAttachments = colorAttachments.data();
  if (depthAttachment.has_value()) renderingInfo.pDepthAttachment = &depthAttachment.value();

  for (auto&& graphElement : _graphElements) graphElement->update(currentFrame, commandBuffer);
  vkCmdBeginRendering(commandBuffer.getCommandBuffer(), &renderingInfo);
  for (auto&& graphElement : _graphElements) graphElement->draw(currentFrame, commandBuffer);
  vkCmdEndRendering(commandBuffer.getCommandBuffer());
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
  _timestamps = std::make_unique<Timestamps>(device, static_cast<uint32_t>(maxFramesInFlight));
  _graphStorage = std::make_unique<GraphStorage>();
  _maxFramesInFlight = maxFramesInFlight;
}

void Graph::initialize() noexcept {
  // create 3 special semaphores
  // Image-available semaphores are indexed by frame-in-flight slot,
  // because they are not tied to a specific swapchain image.
  std::ranges::generate_n(std::back_inserter(_semaphoreImageAvailable), _maxFramesInFlight,
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
  std::ranges::generate_n(std::back_inserter(_semaphoreRenderFinished), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
  _semaphoreInFlight = std::make_unique<Semaphore>(VK_SEMAPHORE_TYPE_TIMELINE, *_device);
}

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

void Graph::Resources::add(Resource resource) {
  auto it = std::find_if(_resources.begin(), _resources.end(),
                         [&resource](const Resource& current) { return current.name == resource.name; });

  if (it == _resources.end()) {
    _resources.push_back(resource);
    return;
  }

  if (it->type != resource.type) {
    throw std::logic_error("Resource with the same name has a different type: " + resource.name);
  }

  it->operation |= resource.operation;
}

bool Graph::Resources::contains(std::string_view name, Resource::Type type, Resource::Operation operation) const {
  const uint8_t operationMask = static_cast<uint8_t>(operation);
  return std::ranges::any_of(_resources, [&](const Resource& resource) {
    return resource.name == name && resource.type == type && (resource.operation & operationMask) != 0;
  });
}

const std::vector<Graph::Resource>& Graph::Resources::getResources() const noexcept { return _resources; }

std::vector<Graph::Resource> Graph::Resources::getResources(Graph::Resource::Operation operation) const {
  const auto operationMask = static_cast<uint8_t>(operation);
  std::vector<Resource> result;
  result.reserve(_resources.size());
  for (auto&& resource : _resources) {
    if ((resource.operation & operationMask) != 0) {
      result.push_back(resource);
    }
  }
  return result;
}

std::vector<std::string> Graph::Resources::getNames(Graph::Resource::Type type) const {
  std::vector<std::string> result;
  result.reserve(_resources.size());

  for (const Resource& resource : _resources) {
    if (resource.type == type) {
      result.push_back(resource.name);
    }
  }

  return result;
}

void Graph::Sync::addSignalSemaphore(std::vector<std::shared_ptr<Semaphore>>& signalSemaphore,
                                     std::function<int()> index) noexcept {
  _signalSemaphores.emplace_back(signalSemaphore, index);
}

void Graph::Sync::addWaitSemaphore(std::vector<std::shared_ptr<Semaphore>>& waitSemaphore,
                                   std::function<int()> index) noexcept {
  _waitSemaphores.emplace_back(waitSemaphore, index);
}

void Graph::Sync::addBarrierBefore(std::vector<Barrier>& barriers, std::function<int()> index) noexcept {
  _barriersBefore.emplace_back(barriers, index);
}

void Graph::Sync::addBarrierAfter(std::vector<Barrier>& barriers, std::function<int()> index) noexcept {
  _barriersAfter.emplace_back(barriers, index);
}

std::vector<Semaphore*> Graph::Sync::getSignalSemaphores() const noexcept {
  return _signalSemaphores | std::views::transform([](auto& pair) {
           auto& [semaphores, index] = pair;
           return semaphores[index()].get();
         }) |
         std::ranges::to<std::vector>();
}

std::vector<Semaphore*> Graph::Sync::getWaitSemaphores() const noexcept {
  return _waitSemaphores | std::views::transform([](auto& pair) {
           auto& [semaphores, index] = pair;
           return semaphores[index()].get();
         }) |
         std::ranges::to<std::vector>();
}

std::vector<const Barrier*> Graph::Sync::getBarriersBefore() const noexcept {
  return _barriersBefore | std::views::transform([](const auto& pair) {
           const auto& [barriers, index] = pair;
           return &barriers[index()];
         }) |
         std::ranges::to<std::vector>();
}

std::vector<const Barrier*> Graph::Sync::getBarriersAfter() const noexcept {
  return _barriersAfter | std::views::transform([](const auto& pair) {
           const auto& [barriers, index] = pair;
           return &barriers[index()];
         }) |
         std::ranges::to<std::vector>();
}

void Graph::print() const noexcept {
  if (_passesOrdered.empty()) return;

  auto findBufferName = [this](VkBuffer targetBuffer) -> std::string {
    for (const auto& [pass, resources] : _resources) {
      for (const Resource& resource : resources.getResources()) {
        if (resource.type != Resource::Type::BUFFER) {
          continue;
        }
        if (!_graphStorage->containsBuffer(resource.name)) {
          continue;
        }
        for (const auto* buffer : _graphStorage->getBuffer(resource.name)) {
          if (buffer->getBuffer() == targetBuffer) {
            return resource.name;
          }
        }
      }
    }

    return "<unknown>";
  };

  auto findImageName = [this](VkImage targetImage) -> std::string {
    for (const auto& [pass, resources] : _resources) {
      for (const Resource& resource : resources.getResources()) {
        if (resource.type != Resource::Type::IMAGE) {
          continue;
        }
        if (!_graphStorage->containsImageViewHolder(resource.name)) {
          continue;
        }
        const auto& imageViews = _graphStorage->getImageViewHolder(resource.name).getImageViews();
        for (const auto& imageView : imageViews) {
          if (imageView->getImage().getImage() == targetImage) {
            return resource.name;
          }
        }
      }
    }

    return "<unknown>";
  };

  auto printImages = [this](const auto& keys, std::string_view keyTag) {
    for (const auto& name : keys) {
      std::cout << keyTag << name << "; ";
      if (!_graphStorage->containsImageViewHolder(name)) {
        std::cout << "<not registered>" << std::endl;
        continue;
      }
      for (const auto& imageView : _graphStorage->getImageViewHolder(name).getImageViews()) {
        std::cout << imageView->getImage().getImage() << " ";
      }
      std::cout << std::endl;
    }
  };

  auto printBuffers = [this](const auto& keys, std::string_view keyTag) {
    for (const auto& name : keys) {
      std::cout << keyTag << name << "; ";
      if (!_graphStorage->containsBuffer(name)) {
        std::cout << "<not registered>" << std::endl;
        continue;
      }
      const auto buffers = _graphStorage->getBuffer(name);
      if (buffers.empty()) {
        std::cout << "<empty>" << std::endl;
        continue;
      }
      for (const auto* buffer : buffers) {
        std::cout << buffer->getBuffer() << " ";
      }
      std::cout << std::endl;
    }
  };

  auto printRange = [](std::string_view label, const auto& range, auto getter) {
    std::cout << label;
    if (std::ranges::empty(range)) {
      std::cout << "<none>" << std::endl;
      return;
    }
    bool first = true;
    for (auto&& item : range) {
      if (!first) {
        std::cout << " ";
      }
      std::cout << getter(item);
      first = false;
    }
    std::cout << std::endl;
  };

  auto accessToString = [](VkAccessFlags2 accessMask) -> std::string_view {
    const bool read = (accessMask & static_cast<VkAccessFlags2>(VK_ACCESS_MEMORY_READ_BIT)) != 0;
    const bool write = (accessMask & static_cast<VkAccessFlags2>(VK_ACCESS_MEMORY_WRITE_BIT)) != 0;
    if (read && write) return "RW";
    if (read) return "R";
    if (write) return "W";

    return "0";
  };

  auto ownershipOperation = [](VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask) -> std::string_view {
    if (srcAccessMask != 0 && dstAccessMask == 0) {
      return "release";
    }
    if (srcAccessMask == 0 && dstAccessMask != 0) {
      return "acquire";
    }
    return "transfer";
  };

  auto printBarrier = [&](const Barrier* barrier, std::size_t barrierIndex) {
    if (barrier == nullptr) {
      std::cout << "  [" << barrierIndex << "] <null>" << std::endl;
      return;
    }

    for (const auto& bufferBarrier : barrier->getBufferBarriers()) {
      const bool ownershipTransfer = bufferBarrier.srcQueueFamilyIndex != std::numeric_limits<uint32_t>::max() ||
                                     bufferBarrier.dstQueueFamilyIndex != std::numeric_limits<uint32_t>::max();
      std::cout << "  [" << barrierIndex << "] "
                << "buffer " << findBufferName(bufferBarrier.buffer) << "[frame " << _frameInFlight << "] ("
                << bufferBarrier.buffer << ") ";
      if (ownershipTransfer) {
        std::cout << "ownership " << ownershipOperation(bufferBarrier.srcAccessMask, bufferBarrier.dstAccessMask)
                  << " ";
      } else {
        std::cout << "memory ";
      }
      std::cout << accessToString(bufferBarrier.srcAccessMask) << " -> " << accessToString(bufferBarrier.dstAccessMask);
      if (ownershipTransfer) {
        std::cout << ", queue " << bufferBarrier.srcQueueFamilyIndex << " -> " << bufferBarrier.dstQueueFamilyIndex;
      }
      std::cout << std::endl;
    }

    for (const auto& imageBarrier : barrier->getImageBarriers()) {
      const bool ownershipTransfer = imageBarrier.srcQueueFamilyIndex != std::numeric_limits<uint32_t>::max() ||
                                     imageBarrier.dstQueueFamilyIndex != std::numeric_limits<uint32_t>::max();
      std::cout << "  [" << barrierIndex << "] "
                << "image " << findImageName(imageBarrier.image) << " (" << imageBarrier.image << ") ";
      if (ownershipTransfer) {
        std::cout << "ownership " << ownershipOperation(imageBarrier.srcAccessMask, imageBarrier.dstAccessMask) << " ";
      } else {
        std::cout << "memory ";
      }
      std::cout << accessToString(imageBarrier.srcAccessMask) << " -> " << accessToString(imageBarrier.dstAccessMask);
      if (ownershipTransfer) {
        std::cout << ", queue " << imageBarrier.srcQueueFamilyIndex << " -> " << imageBarrier.dstQueueFamilyIndex;
      }
      if (imageBarrier.oldLayout != imageBarrier.newLayout) {
        std::cout << ", layout " << static_cast<int>(imageBarrier.oldLayout) << " -> "
                  << static_cast<int>(imageBarrier.newLayout);
      }
      std::cout << std::endl;
    }
  };

  auto printBarriers = [&](std::string_view label, const auto& barriers) {
    std::cout << label;
    if (std::ranges::empty(barriers)) {
      std::cout << "<none>" << std::endl;
      return;
    }
    std::cout << std::endl;
    for (std::size_t index = 0; index < barriers.size(); ++index) {
      printBarrier(barriers[index], index);
    }
  };

  for (auto* value : _passesOrdered) {
    std::cout << "========================================" << std::endl;
    std::cout << "Name: " << value->getName()
              << ", Stage: " << (value->getGraphPassType() == GraphPassType::GRAPHIC ? "GRAPHIC" : "COMPUTE")
              << std::endl;
    if (value->getGraphPassType() == GraphPassType::COMPUTE) {
      const auto* computePass = static_cast<GraphPassCompute*>(value);
      std::cout << " separate: " << (computePass->isSeparate() ? "true" : "false") << std::endl;
    }
    const auto syncIt = _sync.find(value);
    if (syncIt != _sync.end()) {
      const auto& sync = syncIt->second;
      printRange(" wait semaphores: ", sync.getWaitSemaphores(),
                 [](const auto* semaphore) { return semaphore->getSemaphore(); });
      printRange(" signal semaphores: ", sync.getSignalSemaphores(),
                 [](const auto* semaphore) { return semaphore->getSemaphore(); });
      printBarriers(" barriers before: ", sync.getBarriersBefore());
      printBarriers(" barriers after: ", sync.getBarriersAfter());
    } else {
      std::cout << " wait semaphores: <none>" << std::endl;
      std::cout << " signal semaphores: <none>" << std::endl;
      std::cout << " barriers before: <none>" << std::endl;
      std::cout << " barriers after: <none>" << std::endl;
    }

    printRange(" command buffers: ", value->getCommandBuffers(),
               [](const auto* commandBuffer) { return commandBuffer->getCommandBuffer(); });
    if (value->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto* passGraphic = static_cast<GraphPassGraphic*>(value);
      printImages(passGraphic->getColorTargets(), " color target: ");
      const auto& depthTarget = passGraphic->getDepthTarget();
      if (depthTarget) {
        std::cout << " depth target: " << *depthTarget << "; ";
        if (_graphStorage->containsImageViewHolder(*depthTarget)) {
          std::cout << _graphStorage->getImageViewHolder(*depthTarget).getImageView().getImage().getImage();
        } else {
          std::cout << "<not registered>";
        }
        std::cout << std::endl;
      }
      printImages(passGraphic->getTextureInputs(), " texture input: ");
    }

    if (value->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* passCompute = static_cast<GraphPassCompute*>(value);
      printBuffers(passCompute->getStorageBufferInputs(), " storage buffer input: ");
      printBuffers(passCompute->getStorageBufferOutputs(), " storage buffer output: ");
      printImages(passCompute->getStorageTextureInputs(), " storage texture input: ");
      printImages(passCompute->getStorageTextureOutputs(), " storage texture output: ");
    }
  }

  std::cout << "========================================" << std::endl;
}

void Graph::calculate() {
  // store resources in a convinient way
  _resources.clear();
  _passesOrdered.clear();
  _sync.clear();

  auto addResources = [this](GraphPass* pass, const auto& names, Resource::Type type, Resource::Operation operation) {
    for (const auto& name : names) {
      _resources[pass].add({
          .name = name,
          .type = type,
          .operation = static_cast<uint8_t>(operation),
      });
    }
  };

  for (auto&& pass : _passes) {
    if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* compute = static_cast<GraphPassCompute*>(pass.get());
      addResources(pass.get(), compute->getStorageBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), compute->getStorageTextureInputs(), Resource::Type::IMAGE, Resource::Operation::READ);
      addResources(pass.get(), compute->getStorageBufferOutputs(), Resource::Type::BUFFER, Resource::Operation::WRITE);
      addResources(pass.get(), compute->getStorageTextureOutputs(), Resource::Type::IMAGE, Resource::Operation::WRITE);
    } else if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto* graphic = static_cast<GraphPassGraphic*>(pass.get());
      addResources(pass.get(), graphic->getTextureInputs(), Resource::Type::IMAGE, Resource::Operation::READ);
      addResources(pass.get(), graphic->getColorTargets(), Resource::Type::IMAGE, Resource::Operation::WRITE);
      if (const auto& depth = graphic->getDepthTarget()) {
        _resources[pass.get()].add({
            .name = *depth,
            .type = Resource::Type::IMAGE,
            .operation = static_cast<uint8_t>(Resource::Operation::WRITE),
        });
      }
    }
  }

  const auto passes = _passes | std::views::transform([](const auto& pass) { return pass.get(); }) |
                      std::ranges::to<std::vector<GraphPass*>>();
  auto findTarget = [this, &passes](GraphPass* consumer, const Resource& resource) -> GraphPass* {
    const auto consumerIt = std::ranges::find(passes, consumer);
    // Search only among passes located before the consumer.
    for (auto it = std::make_reverse_iterator(consumerIt); it != passes.rend(); ++it) {
      GraphPass* candidate = *it;
      if (_resources.at(candidate).contains(resource.name, resource.type, Resource::Operation::WRITE)) {
        return candidate;
      }
    }
    return nullptr;
  };

  GraphPass* root = _passes.back().get();
  std::unordered_set<GraphPass*> visited;
  std::function<void(GraphPass*)> traverse = [&](GraphPass* node) {
    if (!visited.insert(node).second) {
      return;
    }

    auto resources = _resources.at(node).getResources(Resource::Operation::READ);
    // Special case for the final pass which only writes
    // to the swapchain.
    if (node == root && resources.empty()) {
      resources = _resources.at(node).getResources();
    }

    std::vector<GraphPass*> producers;
    for (const Resource& resource : resources) {
      GraphPass* producer = findTarget(node, resource);
      if (producer != nullptr && !std::ranges::contains(producers, producer)) {
        producers.push_back(producer);
      }
    }

    for (GraphPass* producer : producers) {
      traverse(producer);
    }
    // Add the consumer only after all its producers.
    _passesOrdered.push_back(node);
  };

  // Start from the last node.
  traverse(root);

  // set semaphores between passes + fill barriers
  struct LastUsage {
    GraphPass* pass;
    Resource resource;
  };
  std::unordered_map<std::string, LastUsage> lastImageUsage;
  std::unordered_map<std::string, LastUsage> lastBufferUsage;
  std::unordered_map<std::string, LastUsage> imageOwnership;
  std::unordered_map<std::string, LastUsage> bufferOwnership;

  auto hasOperation = [](const Resource& resource, Resource::Operation operation) {
    return (resource.operation & static_cast<uint8_t>(operation)) != 0;
  };

  auto getAccessMask = [&](const Resource& resource) -> VkAccessFlags2 {
    VkAccessFlags2 accessMask = 0;
    if (hasOperation(resource, Resource::Operation::READ)) {
      accessMask |= static_cast<VkAccessFlags2>(VK_ACCESS_MEMORY_READ_BIT);
    }
    if (hasOperation(resource, Resource::Operation::WRITE)) {
      accessMask |= static_cast<VkAccessFlags2>(VK_ACCESS_MEMORY_WRITE_BIT);
    }

    return accessMask;
  };

  auto getStageMask = [](GraphPass* pass) -> VkPipelineStageFlags2 {
    if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
      return static_cast<VkPipelineStageFlags2>(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    return static_cast<VkPipelineStageFlags2>(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
  };

  auto getImageAspect = [&](std::string_view resourceName) -> VkImageAspectFlags {
    for (GraphPass* pass : _passesOrdered) {
      if (pass->getGraphPassType() != GraphPassType::GRAPHIC) {
        continue;
      }

      auto* graphicPass = static_cast<GraphPassGraphic*>(pass);
      const auto depthTarget = graphicPass->getDepthTarget();
      if (depthTarget && depthTarget.value() == resourceName) {
        return VK_IMAGE_ASPECT_DEPTH_BIT;
      }
    }

    return VK_IMAGE_ASPECT_COLOR_BIT;
  };

  // Creates barrier array for a resource and stores it in the corresponding Sync object.
  auto addResourceBarrier = [&](GraphPass* barrierPass, bool beforePass, const Resource& resource,
                                VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
                                VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask,
                                uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex) {
    if (resource.type == Resource::Type::BUFFER) {
      const auto buffers = _graphStorage->getBuffer(resource.name);
      std::vector<Barrier> barriers(_maxFramesInFlight);
      for (int frameIndex = 0; frameIndex < _maxFramesInFlight; ++frameIndex) {
        barriers[frameIndex].addBuffer(buffers[frameIndex], srcStageMask, srcAccessMask, dstStageMask, dstAccessMask,
                                       srcQueueFamilyIndex, dstQueueFamilyIndex, 0, buffers[frameIndex]->getSize());
      }

      if (beforePass) {
        _sync[barrierPass].addBarrierBefore(barriers, [this]() { return _frameInFlight; });
      } else {
        _sync[barrierPass].addBarrierAfter(barriers, [this]() { return _frameInFlight; });
      }

      return;
    }

    const auto& imageHolder = _graphStorage->getImageViewHolder(resource.name);
    const auto imageViews = imageHolder.getImageViews();
    std::vector<Barrier> barriers(imageViews.size());
    for (std::size_t imageIndex = 0; imageIndex < imageViews.size(); ++imageIndex) {
      auto& image = imageViews[imageIndex]->getImage();
      const VkImageSubresourceRange subresourceRange{
          image.getAspectMask(),
          0,
          static_cast<uint32_t>(image.getMipMapNumber()),
          0,
          static_cast<uint32_t>(image.getLayerNumber()),
      };
      barriers[imageIndex].addImage(image.getImage(), srcStageMask, srcAccessMask, dstStageMask, dstAccessMask,
                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange,
                                    srcQueueFamilyIndex, dstQueueFamilyIndex);
    }

    auto indexFunction = imageHolder.getIndexFunction();
    if (beforePass) {
      _sync[barrierPass].addBarrierBefore(barriers, indexFunction);
    } else {
      _sync[barrierPass].addBarrierAfter(barriers, indexFunction);
    }
  };

  auto getQueueFamilyIndex = [this](GraphPass* pass) -> uint32_t {
    const auto queueType = _usesSeparateQueue(pass) ? vkb::QueueType::compute : vkb::QueueType::graphics;

    return static_cast<uint32_t>(_device->getQueueIndex(queueType));
  };

  bool flagWaitForSwapchain = true;
  GraphPass* previousPass = nullptr;
  for (auto&& pass : _passesOrdered) {
    const bool queueTypeChange = previousPass && _usesSeparateQueue(previousPass) != _usesSeparateQueue(pass);
    if (queueTypeChange) {
      // signal semaphore for the previous pass
      // wait semaphore for the current pass
      std::vector<std::shared_ptr<Semaphore>> semaphoreQueueType(_maxFramesInFlight);
      std::ranges::generate(semaphoreQueueType,
                            [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
      _sync[pass].addWaitSemaphore(semaphoreQueueType, [this]() { return _frameInFlight; });
      _sync[previousPass].addSignalSemaphore(semaphoreQueueType, [this]() { return _frameInFlight; });

      // clear all barriers because of semaphore
      lastImageUsage.clear();
      lastBufferUsage.clear();
    }

    // add barriers
    for (const Resource& resource : _resources.at(pass).getResources()) {
      auto& lastUsage = resource.type == Resource::Type::IMAGE ? lastImageUsage : lastBufferUsage;
      auto& ownership = resource.type == Resource::Type::IMAGE ? imageOwnership : bufferOwnership;
      // Queue-family ownership transfer.
      const auto ownerIt = ownership.find(resource.name);
      if (ownerIt != ownership.end()) {
        const LastUsage& owner = ownerIt->second;
        const uint32_t srcQueueFamilyIndex = getQueueFamilyIndex(owner.pass);
        const uint32_t dstQueueFamilyIndex = getQueueFamilyIndex(pass);
        if (srcQueueFamilyIndex != dstQueueFamilyIndex) {
          // RELEASE:
          // executed after the last source-family use.
          addResourceBarrier(owner.pass, false, owner.resource, getStageMask(owner.pass), getAccessMask(owner.resource),
                             0, 0, srcQueueFamilyIndex, dstQueueFamilyIndex);

          // ACQUIRE:
          // executed before the first destination-family use.
          addResourceBarrier(pass, true, resource, 0, 0, getStageMask(pass), getAccessMask(resource),
                             srcQueueFamilyIndex, dstQueueFamilyIndex);
        }
      }

      // The current pass becomes the latest owner/use.
      ownership[resource.name] = {
          .pass = pass,
          .resource = resource,
      };

      // Normal same-queue hazards.
      const auto previousUsage = lastUsage.find(resource.name);
      if (previousUsage != lastUsage.end()) {
        const LastUsage& previous = previousUsage->second;
        const bool previousWrites = hasOperation(previous.resource, Resource::Operation::WRITE);
        const bool currentWrites = hasOperation(resource, Resource::Operation::WRITE);

        // READ -> READ needs no memory barrier.
        if (previousWrites || currentWrites) {
          addResourceBarrier(pass, true, resource, getStageMask(previous.pass), getAccessMask(previous.resource),
                             getStageMask(pass), getAccessMask(resource), std::numeric_limits<uint32_t>::max(),
                             std::numeric_limits<uint32_t>::max());
        }
      }

      lastUsage[resource.name] = {
          .pass = pass,
          .resource = resource,
      };
    }

    // first pass interacting with the swapchain waits for it
    if (flagWaitForSwapchain) {
      bool swapchainFound = false;
      for (const auto& name : _resources.at(pass).getNames(Resource::Type::IMAGE)) {
        if (_graphStorage->getImageViewHolder(name).contains(_swapchain->getImageViews())) {
          swapchainFound = true;
          break;
        }
      }
      if (swapchainFound) {
        _sync[pass].addWaitSemaphore(_semaphoreImageAvailable, [this]() { return _frameInFlight; });
        flagWaitForSwapchain = false;
      }
    }

    // end node should signal end semaphore
    if (pass == root) {
      _sync[pass].addSignalSemaphore(_semaphoreRenderFinished, [this]() { return _swapchain->getSwapchainIndex(); });
    }

    previousPass = pass;
  }
}

bool Graph::render() {
  // timeline semaphore instead of fence
  if (_valueSemaphoreInFlight > _maxFramesInFlight) {
    uint64_t waitValue = _valueSemaphoreInFlight - _maxFramesInFlight;
    auto semaphoreInFlight = _semaphoreInFlight->getSemaphore();

    VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &semaphoreInFlight,
        .pValues = &waitValue,
    };
    auto result = vkWaitSemaphores(_device->getLogicalDevice(), &waitInfo, std::numeric_limits<std::uint64_t>::max());
    if (result != VK_SUCCESS) {
      throw std::runtime_error("vkWaitSemaphores failed: " + std::to_string(result));
    }
  }

  auto status = _swapchain->acquireNextImage(*_semaphoreImageAvailable[_frameInFlight]);
  if (status == VK_ERROR_OUT_OF_DATE_KHR) {
    return true;
  }
  if (status != VK_SUCCESS && status != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image");
  }

  auto swapchainIndex = _swapchain->getSwapchainIndex();
  _timestamps->resetQueryPool();
  // Record all pass command buffers.
  std::vector<std::future<void>> futureTasks =
      _passesOrdered | std::views::transform([this](GraphPass* pass) {
        auto commandBuffer = pass->getCommandBuffers()[_frameInFlight];
        if (!commandBuffer->getActive()) {
          commandBuffer->beginCommands();
        }

        // Change layouts required for dynamic rendering.
        if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
          auto* passGraphic = static_cast<GraphPassGraphic*>(pass);
          for (const auto& colorTarget : passGraphic->getColorTargets()) {
            auto& imageView = _graphStorage->getImageViewHolder(colorTarget).getImageView();
            auto& image = imageView.getImage();
            if (image.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
              const auto dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
              image.changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, dstAccessMask, *commandBuffer);
            }
          }
          if (const auto& depthTarget = passGraphic->getDepthTarget()) {
            auto& imageView = _graphStorage->getImageViewHolder(*depthTarget).getImageView();
            auto& image = imageView.getImage();
            if (image.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
              const auto dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
              image.changeLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, dstAccessMask, *commandBuffer);
            }
          }
        }

        // execute and add barriers
        // Passes without barriers or semaphores are absent from _sync.
        // Using _sync.at(pass) would throw, so missing entries use an empty Sync.
        Sync sync;
        if (const auto syncIt = _sync.find(pass); syncIt != _sync.end()) {
          sync = syncIt->second;
        }
        return _threadPool->submit([this, pass, commandBuffer, sync]() {
          _timestamps->pushTimestamp(pass->getName(), *commandBuffer);
          for (auto&& barrier : sync.getBarriersBefore()) {
            barrier->execute(commandBuffer->getCommandBuffer());
          }
          pass->execute(_frameInFlight, *commandBuffer);
          for (auto&& barrier : sync.getBarriersAfter()) {
            barrier->execute(commandBuffer->getCommandBuffer());
          }
          _timestamps->popTimestamp(pass->getName(), *commandBuffer);
        });
      }) |
      std::ranges::to<std::vector<std::future<void>>>();

  auto submitPassToQueue = [this](GraphPass* previousPass, const std::vector<CommandBuffer*>& commandBufferSubmit,
                                  const std::vector<VkSemaphore>& waitSemaphores,
                                  const std::vector<VkSemaphore>& signalSemaphores,
                                  const std::optional<std::vector<uint64_t>> signalValues = std::nullopt) {
    std::vector<VkCommandBufferSubmitInfo> commandBufferInfos;
    commandBufferInfos.reserve(commandBufferSubmit.size());
    for (CommandBuffer* commandBuffer : commandBufferSubmit) {
      commandBuffer->endCommands();
      commandBufferInfos.push_back({
          .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
          .commandBuffer = commandBuffer->getCommandBuffer(),
      });
    }

    std::vector<VkSemaphoreSubmitInfo> waitSemaphoreInfos;
    waitSemaphoreInfos.reserve(waitSemaphores.size());
    for (VkSemaphore semaphore : waitSemaphores) {
      waitSemaphoreInfos.push_back({
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = semaphore,
          .value = 0,
          .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      });
    }
    std::vector<VkSemaphoreSubmitInfo> signalSemaphoreInfos;
    signalSemaphoreInfos.reserve(signalSemaphores.size());
    for (std::size_t i = 0; i < signalSemaphores.size(); ++i) {
      signalSemaphoreInfos.push_back({
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = signalSemaphores[i],
          .value = signalValues ? signalValues->at(i) : 0,
          .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      });
    }

    auto queueType = vkb::QueueType::graphics;
    if (previousPass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* passCompute = static_cast<GraphPassCompute*>(previousPass);
      if (passCompute->isSeparate()) {
        queueType = vkb::QueueType::compute;
      }
    }

    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(waitSemaphoreInfos.size()),
        .pWaitSemaphoreInfos = waitSemaphoreInfos.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(commandBufferInfos.size()),
        .pCommandBufferInfos = commandBufferInfos.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphoreInfos.size()),
        .pSignalSemaphoreInfos = signalSemaphoreInfos.data(),
    };

    auto result = vkQueueSubmit2(_device->getQueue(queueType), 1, &submitInfo, nullptr);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("vkQueueSubmit2 failed: " + std::to_string(result));
    }
  };

  std::vector<CommandBuffer*> commandBufferSubmit;
  std::vector<VkSemaphore> signalSemaphores;
  std::vector<VkSemaphore> waitSemaphores;
  GraphPass* previousPass = nullptr;
  for (auto&& [pass, futureTask] : std::views::zip(_passesOrdered, futureTasks)) {
    if (futureTask.valid()) {
      futureTask.get();
    }

    if (previousPass != nullptr) {
      const bool queueTypeChange = _usesSeparateQueue(previousPass) != _usesSeparateQueue(pass);
      if (queueTypeChange) {
        submitPassToQueue(previousPass, commandBufferSubmit, waitSemaphores, signalSemaphores);
        commandBufferSubmit.clear();
        waitSemaphores.clear();
        signalSemaphores.clear();
      }
    }

    previousPass = pass;
    commandBufferSubmit.push_back(pass->getCommandBuffers()[_frameInFlight]);
    if (const auto syncIt = _sync.find(pass); syncIt != _sync.end()) {
      for (Semaphore* semaphore : syncIt->second.getWaitSemaphores()) {
        waitSemaphores.push_back(semaphore->getSemaphore());
      }
      for (Semaphore* semaphore : syncIt->second.getSignalSemaphores()) {
        signalSemaphores.push_back(semaphore->getSemaphore());
      }
    }
  }

  // Change the swapchain image layout before presentation.
  if (_swapchain->getImage(swapchainIndex).getImageLayout() != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
    _swapchain->getImage(swapchainIndex)
        .changeLayout(_swapchain->getImage(swapchainIndex).getImageLayout(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, *commandBufferSubmit.back());
  }

  std::vector<uint64_t> signalValues(signalSemaphores.size() + 1, 0);
  signalValues[signalSemaphores.size()] = _valueSemaphoreInFlight;
  signalSemaphores.push_back(_semaphoreInFlight->getSemaphore());

  submitPassToQueue(_passesOrdered.back(), commandBufferSubmit, waitSemaphores, signalSemaphores, signalValues);

  _timestamps->finishFrame();

  auto semaphoreRenderFinished = _semaphoreRenderFinished[swapchainIndex]->getSemaphore();

  VkSwapchainKHR swapChains[]{
      _swapchain->getSwapchain(),
  };

  VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &semaphoreRenderFinished,
      .swapchainCount = 1,
      .pSwapchains = swapChains,
      .pImageIndices = &swapchainIndex,
  };

  ++_valueSemaphoreInFlight;

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

  _semaphoreRenderFinished.clear();
  std::ranges::generate_n(std::back_inserter(_semaphoreRenderFinished), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });

  for (auto&& pass : _passesOrdered) {
    pass->reset(_swapchain->getImageViews());
  }

  calculate();
}

bool Graph::_usesSeparateQueue(GraphPass* pass) {
  return pass->getGraphPassType() == GraphPassType::COMPUTE && static_cast<GraphPassCompute*>(pass)->isSeparate();
};
