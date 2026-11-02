module;

#include "BS_thread_pool.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <future>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <volk.h>

module Graph;

using namespace RenderGraph;

void GraphStorage::add(std::string_view name,
                       std::unique_ptr<ImageViewHolder> imageHolder,
                       ImageResizePolicy resizePolicy) {
  _images[std::string(name)] = {.imageViewHolder = std::move(imageHolder), .resizePolicy = resizePolicy};
}

void GraphStorage::add(std::string_view name, std::unique_ptr<BufferHandler> bufferHandler) {
  _buffers[std::string(name)] = std::move(bufferHandler);
}

void GraphStorage::reset(std::vector<std::shared_ptr<ImageView>> oldSwapchain,
                         std::vector<std::shared_ptr<ImageView>> newSwapchain) {
  glm::ivec2 resolution = newSwapchain.front()->getImage().getResolution();
  auto nameSwapchain = find(oldSwapchain);
  if (nameSwapchain.empty() == false) {
    _images.at(nameSwapchain).imageViewHolder->setImageViews(newSwapchain);
  }

  for (auto&& [name, resource] : _images) {
    if (name == nameSwapchain || resource.resizePolicy == ImageResizePolicy::FIXED) {
      continue;
    }
    for (auto* imageView : resource.imageViewHolder->getImageViews()) {
      auto& image = imageView->getImage();
      if (image.getResolution() != resolution) {
        imageView->destroy();
        image.destroy();
        image.createImage(image.getFormat(), resolution, image.getMipMapNumber(), image.getLayerNumber(),
                          image.getAspectMask(), image.getUsageFlags(), image.getCreateFlags());
        imageView->createImageView(imageView->getType(), imageView->getBaseMipMap(), imageView->getBaseArrayLayer());
      }
    }
  }
}

std::string GraphStorage::find(const std::vector<std::shared_ptr<ImageView>>& imageViews) {
  for (auto&& [name, resource] : _images) {
    if (resource.imageViewHolder->contains(imageViews)) return name;
  }
  return std::string{};
}

bool GraphStorage::containsImageViewHolder(std::string_view name) const { return _images.contains(std::string(name)); }

const ImageViewHolder& GraphStorage::getImageViewHolder(std::string_view name) const {
  return *_images.at(std::string(name)).imageViewHolder;
}

bool GraphStorage::containsBufferHandler(std::string_view name) const { return _buffers.contains(std::string(name)); }

const BufferHandler& GraphStorage::getBufferHandler(std::string_view name) const {
  return *_buffers.at(std::string(name));
}

GraphPass::GraphPass(std::string_view name, GraphPassType graphPassType, const GraphStorage& graphStorage)
    : _name(name),
      _graphPassType(graphPassType),
      _graphStorage(&graphStorage) {}

void GraphPass::registerGraphElement(std::shared_ptr<GraphElement> graphElement) {
  _graphElements.push_back(graphElement);
}

std::string GraphPass::getName() const { return _name; }

void GraphPass::reset(const std::vector<std::shared_ptr<RenderGraph::ImageView>>& swapchain) {
  for (auto&& graphElement : _graphElements) {
    graphElement->reset(swapchain);
  }
}

GraphPassType GraphPass::getGraphPassType() const noexcept { return _graphPassType; }

QueueType GraphPass::getQueueType() const noexcept { return _commandPool->getType(); }

std::vector<CommandBuffer*> GraphPass::getCommandBuffers() const {
  return _commandBuffers | std::views::transform([](auto& p) { return p.get(); }) | std::ranges::to<std::vector>();
}

GraphPassGraphic::GraphPassGraphic(std::string_view name,
                                   int maxFramesInFlight,
                                   const GraphStorage& graphStorage,
                                   const Device& device)
    : GraphPass(name, GraphPassType::GRAPHIC, graphStorage) {
  _device = &device;
  _commandPool = std::make_unique<CommandPool>(QueueType::GENERAL, device);
  _commandBuffers.resize(maxFramesInFlight);
  std::ranges::generate(_commandBuffers, [&] { return std::make_unique<CommandBuffer>(*_commandPool, device); });
  _pipelineGraphic = std::make_unique<PipelineGraphic>();
}

void GraphPassGraphic::addColorTarget(std::string_view name,
                                      RenderTargetLoadAction loadAction,
                                      RenderTargetStoreAction storeAction,
                                      VkClearValue clearValue) {
  _colorTargets.emplace_back(name);
  _targetSettings[std::string(name)] = {
      .loadAction = loadAction,
      .storeAction = storeAction,
      .clearValue = clearValue,
  };
}

void GraphPassGraphic::setDepthTarget(std::string_view name,
                                      RenderTargetLoadAction loadAction,
                                      RenderTargetStoreAction storeAction,
                                      VkClearValue clearValue) {
  _depthTarget = name;
  _targetSettings[std::string(name)] = {
      .loadAction = loadAction,
      .storeAction = storeAction,
      .clearValue = clearValue,
  };
}

void GraphPassGraphic::addTextureInput(std::string_view name) { _textureInputs.emplace_back(name); }

void GraphPassGraphic::addVertexBufferInput(std::string_view name) { _vertexBufferInputs.emplace_back(name); }

void GraphPassGraphic::addIndexBufferInput(std::string_view name) { _indexBufferInputs.emplace_back(name); }

void GraphPassGraphic::addStorageBufferInput(std::string_view name) { _storageBufferInputs.emplace_back(name); }

void GraphPassGraphic::addIndirectBufferInput(std::string_view name) { _indirectBufferInputs.emplace_back(name); }

const std::vector<std::string>& GraphPassGraphic::getColorTargets() const noexcept { return _colorTargets; }

std::optional<std::string> GraphPassGraphic::getDepthTarget() const { return _depthTarget; }

RenderTargetLoadAction GraphPassGraphic::getTargetLoadAction(std::string_view name) const {
  return _targetSettings.at(std::string(name)).loadAction;
}

const std::vector<std::string>& GraphPassGraphic::getTextureInputs() const noexcept { return _textureInputs; }

const std::vector<std::string>& GraphPassGraphic::getVertexBufferInputs() const noexcept { return _vertexBufferInputs; }

const std::vector<std::string>& GraphPassGraphic::getIndexBufferInputs() const noexcept { return _indexBufferInputs; }

const std::vector<std::string>& GraphPassGraphic::getStorageBufferInputs() const noexcept {
  return _storageBufferInputs;
}

const std::vector<std::string>& GraphPassGraphic::getIndirectBufferInputs() const noexcept {
  return _indirectBufferInputs;
}

PipelineGraphic& GraphPassGraphic::getPipelineGraphic(const GraphStorage& graphStorage) const {
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
  auto getLoadOp = [](RenderTargetLoadAction action) {
    switch (action) {
      case RenderTargetLoadAction::CLEAR:
        return VK_ATTACHMENT_LOAD_OP_CLEAR;
      case RenderTargetLoadAction::KEEP:
        return VK_ATTACHMENT_LOAD_OP_LOAD;
      case RenderTargetLoadAction::DONT_CARE:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }

    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  };

  auto getStoreOp = [](RenderTargetStoreAction action) {
    switch (action) {
      case RenderTargetStoreAction::STORE:
        return VK_ATTACHMENT_STORE_OP_STORE;
      case RenderTargetStoreAction::DONT_CARE:
        return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  };

  auto createColorAttachment = [this, getLoadOp, getStoreOp](const auto& colorTarget) {
    auto& imageViewHolder = _graphStorage->getImageViewHolder(colorTarget);
    const auto& attachmentSettings = _targetSettings.at(colorTarget);
    VkRenderingAttachmentInfo info{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                   .imageView = imageViewHolder.getImageView().getImageView(),
                                   .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .loadOp = getLoadOp(attachmentSettings.loadAction),
                                   .storeOp = getStoreOp(attachmentSettings.storeAction),
                                   .clearValue = attachmentSettings.clearValue};
    return info;
  };

  std::vector<VkRenderingAttachmentInfo> colorAttachments = _colorTargets |
                                                            std::views::transform(createColorAttachment) |
                                                            std::ranges::to<std::vector>();

  std::optional<VkRenderingAttachmentInfo> depthAttachment = std::nullopt;
  if (_depthTarget.has_value()) {
    auto& target = _depthTarget.value();
    auto& imageViewHolder = _graphStorage->getImageViewHolder(target);
    const auto& attachmentSettings = _targetSettings.at(target);

    depthAttachment = VkRenderingAttachmentInfo{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                                .imageView = imageViewHolder.getImageView().getImageView(),
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                .loadOp = getLoadOp(attachmentSettings.loadAction),
                                                .storeOp = getStoreOp(attachmentSettings.storeAction),
                                                .clearValue = attachmentSettings.clearValue};
  }

  VkRenderingInfo renderingInfo = {};
  renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  renderingInfo.renderArea.offset = {0, 0};
  if (_colorTargets.empty() && !_depthTarget) {
    throw std::logic_error("Graphic pass has no attachments: " + _name);
  }
  glm::ivec2 resolution(std::numeric_limits<int>::max());
  for (const auto& target : _colorTargets) {
    const auto& view = _graphStorage->getImageViewHolder(target).getImageView();
    resolution = glm::min(resolution, view.getImage().getResolution(view.getBaseMipMap()));
  }
  if (_depthTarget) {
    const auto& view = _graphStorage->getImageViewHolder(*_depthTarget).getImageView();
    resolution = glm::min(resolution, view.getImage().getResolution(view.getBaseMipMap()));
  }
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
                                   QueueType queueType,
                                   const GraphStorage& graphStorage,
                                   const Device& device)
    : GraphPass(name, GraphPassType::COMPUTE, graphStorage) {
  if (queueType != QueueType::GENERAL && queueType != QueueType::COMPUTE) {
    throw std::invalid_argument("Compute pass requires a GENERAL or COMPUTE queue: " + std::string(name));
  }
  _device = &device;

  _commandPool = std::make_unique<CommandPool>(queueType, device);
  _commandBuffers.resize(maxFramesInFlight);
  std::ranges::generate(_commandBuffers, [&] { return std::make_unique<CommandBuffer>(*_commandPool, device); });
}

void GraphPassCompute::addStorageBufferInput(std::string_view name) { _storageBufferInputs.emplace_back(name); }

void GraphPassCompute::addStorageTextureInput(std::string_view name) { _storageTextureInputs.emplace_back(name); }

void GraphPassCompute::addIndirectBufferInput(std::string_view name) { _indirectBufferInputs.emplace_back(name); }

void GraphPassCompute::addStorageBufferOutput(std::string_view name) { _storageBufferOutputs.emplace_back(name); }

void GraphPassCompute::addStorageTextureOutput(std::string_view name) { _storageTextureOutputs.emplace_back(name); }

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

const std::vector<std::string>& GraphPassCompute::getIndirectBufferInputs() const noexcept {
  return _indirectBufferInputs;
}

void GraphPassCompute::execute(int currentFrame, const CommandBuffer& commandBuffer) {
  for (auto&& graphElement : _graphElements) {
    graphElement->update(currentFrame, commandBuffer);
    graphElement->draw(currentFrame, commandBuffer);
  }
}

void Graph::_setResourceNames() {
  for (const auto& resourceEntry : _resources) {
    for (const Resource& resource : resourceEntry.second.getResources()) {
      if (resource.type == Resource::Type::BUFFER) {
        const auto buffers = _graphStorage->getBufferHandler(resource.name).getBuffers();
        for (std::size_t index = 0; index < buffers.size(); ++index) {
          _tracer->setObjectName(VK_OBJECT_TYPE_BUFFER, std::bit_cast<uint64_t>(buffers[index]->getBuffer()),
                                 resource.name + ".buffer[" + std::to_string(index) + "]");
        }
      } else {
        const auto imageViews = _graphStorage->getImageViewHolder(resource.name).getImageViews();
        for (std::size_t index = 0; index < imageViews.size(); ++index) {
          const std::string suffix = "[" + std::to_string(index) + "]";
          _tracer->setObjectName(VK_OBJECT_TYPE_IMAGE,
                                 std::bit_cast<uint64_t>(imageViews[index]->getImage().getImage()),
                                 resource.name + ".image" + suffix);
          _tracer->setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, std::bit_cast<uint64_t>(imageViews[index]->getImageView()),
                                 resource.name + ".imageView" + suffix);
        }
      }
    }
  }
}

Graph::Graph(int threadsNumber, int maxFramesInFlight, const Device& device, Tracer& tracer)
    : _device(&device),
      _tracer(&tracer) {
  _threadPool = std::make_unique<BS::thread_pool>(threadsNumber);
  _graphStorage = std::make_unique<GraphStorage>();
  _maxFramesInFlight = maxFramesInFlight;
}

void Graph::setSwapchain(Swapchain& swapchain) {
  if (_swapchain != nullptr) {
    throw std::logic_error("Swapchain is already set");
  }
  if (swapchain.getImageCount() == 0) {
    throw std::logic_error("Can't set an uninitialized swapchain");
  }

  _swapchain = &swapchain;
  std::ranges::generate_n(std::back_inserter(_semaphoreImageAvailable), _maxFramesInFlight,
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
  std::ranges::generate_n(std::back_inserter(_semaphoreRenderFinished), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
}

void Graph::initialize() { _semaphoreInFlight = std::make_unique<Semaphore>(VK_SEMAPHORE_TYPE_TIMELINE, *_device); }

GraphStorage& Graph::getGraphStorage() const noexcept { return *_graphStorage; }

int Graph::getFrameInFlight() const noexcept { return static_cast<int>(getFrameNumber() % _maxFramesInFlight); }

uint64_t Graph::getFrameNumber() const noexcept { return _frameNumber; }

GraphPassGraphic& Graph::createPassGraphic(std::string_view name) {
  auto it = std::find_if(_passes.begin(), _passes.end(),
                         [name = name](std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == name; });
  if (it != _passes.end()) {
    throw std::logic_error("Pass name is already in use: " + std::string(name));
  }

  _passes.push_back(std::make_unique<GraphPassGraphic>(name, _maxFramesInFlight, *_graphStorage, *_device));
  return static_cast<GraphPassGraphic&>(*_passes.back());
}

GraphPassCompute& Graph::createPassCompute(std::string_view name, QueueType queueType) {
  auto it = std::find_if(_passes.begin(), _passes.end(),
                         [name = name](std::unique_ptr<GraphPass>& graphPass) { return graphPass->getName() == name; });
  if (it != _passes.end()) {
    throw std::logic_error("Pass name is already in use: " + std::string(name));
  }

  _passes.push_back(std::make_unique<GraphPassCompute>(name, _maxFramesInFlight, queueType, *_graphStorage, *_device));
  return static_cast<GraphPassCompute&>(*_passes.back());
}

GraphPassGraphic* Graph::getPassGraphic(std::string_view name) const {
  const auto it = std::ranges::find_if(_passes, [name](const auto& pass) {
    return pass->getName() == name && pass->getGraphPassType() == GraphPassType::GRAPHIC;
  });
  return it == _passes.end() ? nullptr : static_cast<GraphPassGraphic*>(it->get());
}

GraphPassCompute* Graph::getPassCompute(std::string_view name) const {
  const auto it = std::ranges::find_if(_passes, [name](const auto& pass) {
    return pass->getName() == name && pass->getGraphPassType() == GraphPassType::COMPUTE;
  });
  return it == _passes.end() ? nullptr : static_cast<GraphPassCompute*>(it->get());
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
                                     std::function<int()> index) {
  _signalSemaphores.emplace_back(signalSemaphore, index);
}

void Graph::Sync::addWaitSemaphore(std::vector<std::shared_ptr<Semaphore>>& waitSemaphore, std::function<int()> index) {
  _waitSemaphores.emplace_back(waitSemaphore, index);
}

void Graph::Sync::addBarrier(std::vector<Barrier>& barriers, std::function<int()> index) {
  _barriers.emplace_back(barriers, index);
}

std::vector<Semaphore*> Graph::Sync::getSignalSemaphores() const {
  return _signalSemaphores | std::views::transform([](auto& pair) {
           auto& [semaphores, index] = pair;
           return semaphores.at(index()).get();
         }) |
         std::ranges::to<std::vector>();
}

std::vector<Semaphore*> Graph::Sync::getWaitSemaphores() const {
  return _waitSemaphores | std::views::transform([](auto& pair) {
           auto& [semaphores, index] = pair;
           return semaphores.at(index()).get();
         }) |
         std::ranges::to<std::vector>();
}

std::vector<const Barrier*> Graph::Sync::getBarriers() const {
  return _barriers | std::views::transform([](const auto& pair) {
           const auto& [barriers, index] = pair;
           return &barriers.at(index());
         }) |
         std::ranges::to<std::vector>();
}

void Graph::print() const {
  if (_passesOrdered.empty()) return;

  auto findBufferName = [this](VkBuffer targetBuffer) -> std::string {
    for (const auto& [pass, resources] : _resources) {
      for (const Resource& resource : resources.getResources()) {
        if (resource.type != Resource::Type::BUFFER) {
          continue;
        }
        if (!_graphStorage->containsBufferHandler(resource.name)) {
          continue;
        }
        for (const auto* buffer : _graphStorage->getBufferHandler(resource.name).getBuffers()) {
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
      if (!_graphStorage->containsBufferHandler(name)) {
        std::cout << "<not registered>" << std::endl;
        continue;
      }
      const auto buffers = _graphStorage->getBufferHandler(name).getBuffers();
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
    const bool read = (accessMask & VK_ACCESS_2_MEMORY_READ_BIT) != 0;
    const bool write = (accessMask & VK_ACCESS_2_MEMORY_WRITE_BIT) != 0;
    if (read && write) return "RW";
    if (read) return "R";
    if (write) return "W";

    return "0";
  };

  auto printBarrier = [&](const Barrier* barrier, std::size_t barrierIndex) {
    if (barrier == nullptr) {
      std::cout << "  [" << barrierIndex << "] <null>" << std::endl;
      return;
    }

    for (const auto& bufferBarrier : barrier->getBufferBarriers()) {
      std::cout << "  [" << barrierIndex << "] "
                << "buffer " << findBufferName(bufferBarrier.buffer) << "[frame " << getFrameInFlight() << "] ("
                << bufferBarrier.buffer << ") memory " << accessToString(bufferBarrier.srcAccessMask) << " -> "
                << accessToString(bufferBarrier.dstAccessMask) << std::endl;
    }
    for (const auto& imageBarrier : barrier->getImageBarriers()) {
      std::cout << "  [" << barrierIndex << "] "
                << "image " << findImageName(imageBarrier.image) << " (" << imageBarrier.image << ") memory "
                << accessToString(imageBarrier.srcAccessMask) << " -> " << accessToString(imageBarrier.dstAccessMask);
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
    std::cout << " queue: " << (value->getQueueType() == QueueType::COMPUTE ? "COMPUTE" : "GENERAL") << std::endl;
    const auto syncIt = _sync.find(value);
    if (syncIt != _sync.end()) {
      const auto& sync = syncIt->second;
      printRange(" wait semaphores: ", sync.getWaitSemaphores(),
                 [](const auto* semaphore) { return semaphore->getSemaphore(); });
      printRange(" signal semaphores: ", sync.getSignalSemaphores(),
                 [](const auto* semaphore) { return semaphore->getSemaphore(); });
      printBarriers(" barriers: ", sync.getBarriers());
    } else {
      std::cout << " wait semaphores: <none>" << std::endl;
      std::cout << " signal semaphores: <none>" << std::endl;
      std::cout << " barriers: <none>" << std::endl;
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
      printBuffers(passGraphic->getVertexBufferInputs(), " vertex buffer input: ");
      printBuffers(passGraphic->getIndexBufferInputs(), " index buffer input: ");
      printBuffers(passGraphic->getStorageBufferInputs(), " storage buffer input: ");
      printBuffers(passGraphic->getIndirectBufferInputs(), " indirect buffer input: ");
    }

    if (value->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* passCompute = static_cast<GraphPassCompute*>(value);
      printBuffers(passCompute->getStorageBufferInputs(), " storage buffer input: ");
      printBuffers(passCompute->getStorageBufferOutputs(), " storage buffer output: ");
      printBuffers(passCompute->getIndirectBufferInputs(), " indirect buffer input: ");
      printImages(passCompute->getStorageTextureInputs(), " storage texture input: ");
      printImages(passCompute->getStorageTextureOutputs(), " storage texture output: ");
    }
  }

  std::cout << "========================================" << std::endl;
}

void Graph::calculate(std::span<const GpuTicket> tickets) {
  const auto cpuTrace = _tracer->traceCPU("Graph.calculate");
  // Tickets from one producer share a timeline semaphore; its highest value covers all its earlier batches.
  for (const auto& ticket : tickets) {
    const auto pending = std::ranges::find(_pendingTickets, ticket.semaphore, &GpuTicket::semaphore);
    if (pending == _pendingTickets.end()) {
      _pendingTickets.push_back(ticket);
    } else {
      pending->value = std::max(pending->value, ticket.value);
    }
  }
  // store resources in a convinient way
  _resources.clear();
  _passesOrdered.clear();
  _passQueueTypes.clear();
  _collectionPasses.clear();
  _sync.clear();
  _physicalAccesses.clear();
  _physicalBarriers.clear();
  _initialImageBarriers.clear();
  _swapchainFirstPass = nullptr;
  _swapchainFirstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  _swapchainFirstAccessMask = 0;

  if (_passes.empty()) {
    return;
  }

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
    // initialize Resources structure for current pass
    _resources.try_emplace(pass.get());
    if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* compute = static_cast<GraphPassCompute*>(pass.get());
      addResources(pass.get(), compute->getStorageBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), compute->getIndirectBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), compute->getStorageTextureInputs(), Resource::Type::IMAGE, Resource::Operation::READ);
      addResources(pass.get(), compute->getStorageBufferOutputs(), Resource::Type::BUFFER, Resource::Operation::WRITE);
      addResources(pass.get(), compute->getStorageTextureOutputs(), Resource::Type::IMAGE, Resource::Operation::WRITE);
    } else if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto* graphic = static_cast<GraphPassGraphic*>(pass.get());
      addResources(pass.get(), graphic->getVertexBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), graphic->getIndexBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), graphic->getStorageBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), graphic->getIndirectBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ);
      addResources(pass.get(), graphic->getTextureInputs(), Resource::Type::IMAGE, Resource::Operation::READ);

      auto getTargetOperation = [graphic](const auto& target) {
        uint8_t operation = static_cast<uint8_t>(Resource::Operation::WRITE);
        // KEEP loads the previous attachment contents, so the target is both read and written.
        if (graphic->getTargetLoadAction(target) == RenderTargetLoadAction::KEEP) {
          operation |= static_cast<uint8_t>(Resource::Operation::READ);
        }
        return static_cast<Resource::Operation>(operation);
      };

      for (const auto& colorTarget : graphic->getColorTargets()) {
        addResources(pass.get(), std::views::single(colorTarget), Resource::Type::IMAGE,
                     getTargetOperation(colorTarget));
      }
      if (const auto& depth = graphic->getDepthTarget()) {
        addResources(pass.get(), std::views::single(*depth), Resource::Type::IMAGE, getTargetOperation(*depth));
      }
    }
  }

  _setResourceNames();

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

  std::unordered_set<GraphPass*> visited;
  std::function<void(GraphPass*)> traverse = [&](GraphPass* node) {
    if (!visited.insert(node).second) {
      return;
    }

    auto resources = _resources.at(node).getResources(Resource::Operation::READ);

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

  for (auto&& pass : _passes) traverse(pass.get());

  // Compile per-name barriers for ordinary instances; physical candidates are excluded below.
  struct ResourceUsage {
    VkPipelineStageFlags2 stageMask = 0;
    VkAccessFlags2 accessMask = 0;
  };

  struct ResourceHistory {
    ResourceUsage lastWrite;
    ResourceUsage readers;
  };

  auto hasOperation = [](const Resource& resource, Resource::Operation operation) {
    return (resource.operation & static_cast<uint8_t>(operation)) != 0;
  };

  auto getResourceUsage = [&](GraphPass* pass, const Resource& resource) -> ResourceUsage {
    ResourceUsage usage;

    if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* compute = static_cast<GraphPassCompute*>(pass);

      if (resource.type == Resource::Type::BUFFER) {
        if (std::ranges::contains(compute->getStorageBufferInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          usage.accessMask |= VK_ACCESS_2_SHADER_READ_BIT;
        }
        if (std::ranges::contains(compute->getStorageBufferOutputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          usage.accessMask |= VK_ACCESS_2_SHADER_WRITE_BIT;
        }
        if (std::ranges::contains(compute->getIndirectBufferInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
          usage.accessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        }
      } else {
        if (std::ranges::contains(compute->getStorageTextureInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          usage.accessMask |= VK_ACCESS_2_SHADER_READ_BIT;
        }
        if (std::ranges::contains(compute->getStorageTextureOutputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          usage.accessMask |= VK_ACCESS_2_SHADER_WRITE_BIT;
        }
      }
    } else if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto* graphic = static_cast<GraphPassGraphic*>(pass);

      if (resource.type == Resource::Type::BUFFER) {
        if (std::ranges::contains(graphic->getVertexBufferInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
          usage.accessMask |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        }
        if (std::ranges::contains(graphic->getIndexBufferInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
          usage.accessMask |= VK_ACCESS_2_INDEX_READ_BIT;
        }
        if (std::ranges::contains(graphic->getStorageBufferInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
          usage.accessMask |= VK_ACCESS_2_SHADER_READ_BIT;
        }
        if (std::ranges::contains(graphic->getIndirectBufferInputs(), resource.name)) {
          usage.stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
          usage.accessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        }
      } else if (std::ranges::contains(graphic->getTextureInputs(), resource.name)) {
        usage.stageMask |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        usage.accessMask |= VK_ACCESS_2_SHADER_READ_BIT;
      }

      if (std::ranges::contains(graphic->getColorTargets(), resource.name)) {
        usage.stageMask |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        if (hasOperation(resource, Resource::Operation::READ)) {
          usage.accessMask |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        }
        if (hasOperation(resource, Resource::Operation::WRITE)) {
          usage.accessMask |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        }
      }

      const auto depthTarget = graphic->getDepthTarget();

      if (depthTarget && *depthTarget == resource.name) {
        usage.stageMask |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

        if (hasOperation(resource, Resource::Operation::READ)) {
          usage.accessMask |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        }
        if (hasOperation(resource, Resource::Operation::WRITE)) {
          usage.accessMask |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
      }
    }

    if (usage.stageMask == 0 || usage.accessMask == 0) {
      throw std::runtime_error("Unable to determine usage of resource " + resource.name + " in pass " +
                               std::string(pass->getName()));
    }

    return usage;
  };

  // Find physical instances with multiple used names and at least one WRITE.
  struct Candidate {
    std::unordered_set<std::string> names;
    bool writes = false;
  };
  std::unordered_map<VkBuffer, Candidate> buffers;
  std::unordered_map<VkImage, Candidate> images;
  for (const auto& [pass, resources] : _resources) {
    for (const auto& resource : resources.getResources()) {
      auto addCandidate = [&](Candidate& candidate) {
        candidate.names.insert(resource.name);
        candidate.writes |= hasOperation(resource, Resource::Operation::WRITE);
      };
      if (resource.type == Resource::Type::BUFFER) {
        for (const auto* buffer : _graphStorage->getBufferHandler(resource.name).getBuffers()) {
          addCandidate(buffers[buffer->getBuffer()]);
        }
      } else {
        for (const auto* view : _graphStorage->getImageViewHolder(resource.name).getImageViews()) {
          addCandidate(images[view->getImage().getImage()]);
        }
      }
    }
  }
  auto isCandidate = [](const Candidate& candidate) { return candidate.names.size() > 1 && candidate.writes; };
  std::erase_if(_bufferHistory, [&](const auto& entry) {
    const auto found = buffers.find(entry.first);
    return found == buffers.end() || !isCandidate(found->second);
  });
  std::erase_if(_imageHistory, [&](const auto& entry) {
    const auto found = images.find(entry.first);
    return found == images.end() || !isCandidate(found->second);
  });
  for (const auto& [buffer, candidate] : buffers) {
    if (isCandidate(candidate)) _bufferHistory.try_emplace(buffer);
  }
  for (const auto& [image, candidate] : images) {
    if (isCandidate(candidate)) _imageHistory.try_emplace(image);
  }

  // Split the used instances once, preserving handler indices for the ordinary per-name barriers.
  std::unordered_map<std::string, std::vector<std::pair<std::size_t, Buffer*>>> ordinaryBuffers;
  std::unordered_map<std::string, std::vector<std::pair<std::size_t, ImageView*>>> ordinaryImages;
  for (const auto& [pass, resources] : _resources) {
    for (const auto& resource : resources.getResources()) {
      if (resource.type == Resource::Type::BUFFER) {
        auto [ordinary, added] = ordinaryBuffers.try_emplace(resource.name);
        if (!added) continue;
        const auto instances = _graphStorage->getBufferHandler(resource.name).getBuffers();
        for (std::size_t index = 0; index < instances.size(); ++index) {
          if (!_bufferHistory.contains(instances[index]->getBuffer())) {
            ordinary->second.emplace_back(index, instances[index]);
          }
        }
      } else {
        auto [ordinary, added] = ordinaryImages.try_emplace(resource.name);
        if (!added) continue;
        const auto instances = _graphStorage->getImageViewHolder(resource.name).getImageViews();
        for (std::size_t index = 0; index < instances.size(); ++index) {
          if (!_imageHistory.contains(instances[index]->getImage().getImage())) {
            ordinary->second.emplace_back(index, instances[index]);
          }
        }
      }
    }
  }

  // Only ordinary instances reach the barriers compiled by name.
  auto addResourceBarrier = [&](GraphPass* barrierPass, std::string_view resourceName, Resource::Type resourceType,
                                VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
                                VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask) {
    if (resourceType == Resource::Type::BUFFER) {
      const auto& bufferHandler = _graphStorage->getBufferHandler(resourceName);
      std::vector<Barrier> barriers(bufferHandler.getBuffers().size());
      for (const auto& [index, buffer] : ordinaryBuffers.at(std::string(resourceName))) {
        barriers[index].addBuffer(buffer, srcStageMask, srcAccessMask, dstStageMask, dstAccessMask, 0,
                                  buffer->getSize());
      }

      _sync[barrierPass].addBarrier(barriers, bufferHandler.getIndexFunction());

      return;
    }

    const auto& imageHolder = _graphStorage->getImageViewHolder(resourceName);
    std::vector<Barrier> barriers(imageHolder.getImageViews().size());
    for (const auto& [index, view] : ordinaryImages.at(std::string(resourceName))) {
      auto& image = view->getImage();
      const VkImageSubresourceRange subresourceRange{
          image.getAspectMask(),
          0,
          static_cast<uint32_t>(image.getMipMapNumber()),
          0,
          static_cast<uint32_t>(image.getLayerNumber()),
      };
      barriers[index].addImage(image.getImage(), srcStageMask, srcAccessMask, dstStageMask, dstAccessMask,
                               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
    }

    auto indexFunction = imageHolder.getIndexFunction();
    _sync[barrierPass].addBarrier(barriers, indexFunction);
  };

  auto makeLayoutBarrier = [](Image& image, VkImageLayout oldLayout, VkImageLayout newLayout, ResourceUsage source,
                              ResourceUsage destination) {
    Barrier barrier;
    barrier.addImage(image.getImage(), source.stageMask, source.accessMask, destination.stageMask,
                     destination.accessMask, oldLayout, newLayout,
                     {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                      static_cast<uint32_t>(image.getLayerNumber())});
    return barrier;
  };
  GraphPass* root = _passes.back().get();
  if (_swapchain != nullptr) {
    // Wait before the first swapchain use, or on the root pass when only its final
    // layout transition touches the acquired image.
    const auto acquireWaitIt = std::ranges::find_if(_passesOrdered, [this](GraphPass* pass) {
      return std::ranges::any_of(_resources.at(pass).getNames(Resource::Type::IMAGE), [this](const auto& name) {
        return _graphStorage->getImageViewHolder(name).contains(_swapchain->getImageViews());
      });
    });
    _swapchainFirstPass = acquireWaitIt != _passesOrdered.end() ? *acquireWaitIt : root;
    _sync[_swapchainFirstPass].addWaitSemaphore(_semaphoreImageAvailable, [this]() { return getFrameInFlight(); });
  }

  // Track resource hazards across passes.
  std::unordered_map<std::string, ResourceHistory> imageHistory;
  std::unordered_map<std::string, ResourceHistory> bufferHistory;
  struct ImageLayoutUsage {
    GraphPass* firstPass = nullptr;
    ResourceUsage usage;
  };
  std::unordered_map<std::string, ImageLayoutUsage> imageLayoutUsage;
  // Select one pass per actual queue so render() doesn't discover queues every frame for traces.
  std::unordered_set<VkQueue> collectionQueues;
  GraphPass* previousPass = nullptr;
  for (auto&& pass : _passesOrdered) {
    const QueueType queueType = pass->getQueueType();
    _passQueueTypes.emplace(pass, queueType);
    if (collectionQueues.insert(_device->getQueue(queueType)).second) {
      _collectionPasses.insert(pass);
    }

    const bool queueTypeChange = previousPass && _passQueueTypes.at(previousPass) != queueType;
    if (queueTypeChange) {
      // signal semaphore for the previous pass
      // wait semaphore for the current pass
      std::vector<std::shared_ptr<Semaphore>> semaphoreQueueType(_maxFramesInFlight);
      std::ranges::generate(semaphoreQueueType,
                            [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
      _sync[pass].addWaitSemaphore(semaphoreQueueType, [this]() { return getFrameInFlight(); });
      _sync[previousPass].addSignalSemaphore(semaphoreQueueType, [this]() { return getFrameInFlight(); });

      // clear all barriers because of semaphore
      imageHistory.clear();
      bufferHistory.clear();
    }

    // add barriers
    for (const Resource& resource : _resources.at(pass).getResources()) {
      const ResourceUsage currentUsage = getResourceUsage(pass, resource);
      // Prepare each pass's candidate selectors, READ/WRITE roles, and stage/access masks.
      PhysicalAccess physical{.stageMask = currentUsage.stageMask,
                              .accessMask = currentUsage.accessMask,
                              .writes = hasOperation(resource, Resource::Operation::WRITE)};
      if (resource.type == Resource::Type::BUFFER) {
        const auto& handler = _graphStorage->getBufferHandler(resource.name);
        if (ordinaryBuffers.at(resource.name).size() < handler.getBuffers().size()) physical.buffer = &handler;
      } else {
        const auto& holder = _graphStorage->getImageViewHolder(resource.name);
        if (ordinaryImages.at(resource.name).size() < holder.getImageViews().size()) physical.image = &holder;
      }
      if (physical.buffer || physical.image) {
        _physicalAccesses[pass].push_back(physical);
        _physicalBarriers.try_emplace(pass);
      }

      if (resource.type == Resource::Type::IMAGE) {
        auto& layout = imageLayoutUsage[resource.name];
        if (layout.firstPass == nullptr) {
          layout.firstPass = pass;
        }
        if (_device->getQueue(queueType) == _device->getQueue(_passQueueTypes.at(layout.firstPass))) {
          layout.usage.stageMask |= currentUsage.stageMask;
          layout.usage.accessMask |= currentUsage.accessMask;
        }
      }

      const bool hasOrdinaryInstances = resource.type == Resource::Type::BUFFER
                                            ? !ordinaryBuffers.at(resource.name).empty()
                                            : !ordinaryImages.at(resource.name).empty();
      if (!hasOrdinaryInstances) continue;
      auto& history = (resource.type == Resource::Type::IMAGE ? imageHistory : bufferHistory)[resource.name];

      // Keep the last writer for W-R-R across different reader stages/accesses.
      // Accumulate all reader masks so R-R-W waits for every reader before overwriting the resource.
      const bool currentWrites = hasOperation(resource, Resource::Operation::WRITE);
      ResourceUsage sourceUsage = history.lastWrite;
      if (currentWrites) {
        sourceUsage.stageMask |= history.readers.stageMask;
        sourceUsage.accessMask |= history.readers.accessMask;
      }

      // srcStageMask — Pipeline stages that must complete before the destination stages proceed.
      // dstStageMask — Pipeline stages that must wait for the source stages.
      // srcAccessMask — Source memory accesses whose writes are made available.
      // dstAccessMask — Destination memory accesses for which those writes become visible.

      if (sourceUsage.stageMask != 0) {
        addResourceBarrier(pass, resource.name, resource.type, sourceUsage.stageMask, sourceUsage.accessMask,
                           currentUsage.stageMask, currentUsage.accessMask);
      }

      if (currentWrites) {
        history.lastWrite = currentUsage;
        history.readers = {};
      } else {
        history.readers.stageMask |= currentUsage.stageMask;
        history.readers.accessMask |= currentUsage.accessMask;
      }
    }

    // end node should signal end semaphore
    if (_swapchain != nullptr && pass == root) {
      _sync[pass].addSignalSemaphore(_semaphoreRenderFinished, [this]() { return _swapchain->getSwapchainIndex(); });
    }

    previousPass = pass;
  }

  for (const auto& [name, layout] : imageLayoutUsage) {
    const auto& holder = _graphStorage->getImageViewHolder(name);
    if (_swapchain != nullptr && holder.contains(_swapchain->getImageViews())) {
      _swapchainFirstStageMask = layout.usage.stageMask;
      _swapchainFirstAccessMask = layout.usage.accessMask;
      continue;
    }
    for (const auto& [index, view] : ordinaryImages.at(name)) {
      auto& image = view->getImage();
      if (image.getImageLayout() == VK_IMAGE_LAYOUT_GENERAL) continue;
      _initialImageBarriers[layout.firstPass].emplace_back(
          &image, makeLayoutBarrier(image, image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0}, layout.usage));
    }
  }
  // Create one timeline per real queue when the graph uses multiple queues and has physical candidates.
  if (collectionQueues.size() > 1 && (!_bufferHistory.empty() || !_imageHistory.empty())) {
    for (const auto queue : collectionQueues) {
      auto [timeline, added] = _queueTimelines.try_emplace(queue);
      if (added) timeline->second.semaphore = std::make_unique<Semaphore>(VK_SEMAPHORE_TYPE_TIMELINE, *_device);
    }
  }
  for (auto& [buffer, history] : _bufferHistory) history.readers.reserve(collectionQueues.size());
  for (auto& [image, history] : _imageHistory) history.readers.reserve(collectionQueues.size());
}

// Merge waits for one semaphore using the highest completion value and all required destination stages.
void Graph::_addWait(std::vector<VkSemaphoreSubmitInfo>& waits,
                     VkSemaphore semaphore,
                     uint64_t value,
                     VkPipelineStageFlags2 stages) {
  auto wait = std::ranges::find(waits, semaphore, &VkSemaphoreSubmitInfo::semaphore);
  if (wait != waits.end()) {
    wait->value = std::max(wait->value, value);
    wait->stageMask |= stages;
  } else {
    waits.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                     .semaphore = semaphore,
                     .value = value,
                     .stageMask = stages});
  }
}

void Graph::_preparePhysicalAccesses(std::unordered_map<GraphPass*, uint64_t>& submissionValues,
                                     std::unordered_map<GraphPass*, std::vector<VkSemaphoreSubmitInfo>>& waits) {
  // Assign one timeline value to each existing submission, shared by all passes in that group.
  GraphPass* previousPass = nullptr;
  uint64_t submissionValue = 0;
  for (auto* pass : _passesOrdered) {
    if (!previousPass || _passQueueTypes.at(previousPass) != _passQueueTypes.at(pass)) {
      const auto queue = _device->getQueue(_passQueueTypes.at(pass));
      if (auto timeline = _queueTimelines.find(queue); timeline != _queueTimelines.end()) {
        submissionValue = ++timeline->second.value;
      }
    }
    submissionValues.emplace(pass, submissionValue);
    previousPass = pass;
  }
  // Combine aliases of one selected instance into one READ/WRITE history update per pass.
  struct ResolvedAccess {
    PhysicalHistory* history;
    Buffer* buffer = nullptr;
    Image* image = nullptr;
    VkPipelineStageFlags2 stageMask;
    VkAccessFlags2 accessMask;
    bool writes;
  };
  std::vector<ResolvedAccess> resolved;
  const VkImage acquiredImage = _swapchain ? _swapchain->getImage(_swapchain->getSwapchainIndex()).getImage()
                                           : VK_NULL_HANDLE;
  for (auto* pass : _passesOrdered) {
    const auto queue = _device->getQueue(_passQueueTypes.at(pass));
    const auto value = submissionValues.at(pass);
    const auto physical = _physicalAccesses.find(pass);
    if (physical == _physicalAccesses.end()) continue;
    const auto& accesses = physical->second;
    auto& barrier = _physicalBarriers.at(pass);
    barrier.clear();
    resolved.clear();
    resolved.reserve(accesses.size());
    for (const auto& access : accesses) {
      ResolvedAccess current{.stageMask = access.stageMask, .accessMask = access.accessMask, .writes = access.writes};
      if (access.buffer) {
        current.buffer = &access.buffer->getBuffer();
        auto history = _bufferHistory.find(current.buffer->getBuffer());
        if (history == _bufferHistory.end()) continue;
        current.history = &history->second;
      } else {
        current.image = &access.image->getImageView().getImage();
        auto history = _imageHistory.find(current.image->getImage());
        if (history == _imageHistory.end()) continue;
        current.history = &history->second;
      }
      auto existing = std::ranges::find(resolved, current.history, &ResolvedAccess::history);
      if (existing == resolved.end()) {
        resolved.push_back(current);
      } else {
        existing->stageMask |= current.stageMask;
        existing->accessMask |= current.accessMask;
        existing->writes |= current.writes;
      }
    }
    for (const auto& access : resolved) {
      auto& history = *access.history;
      const bool transition = access.image && access.image->getImage() != acquiredImage &&
                              access.image->getImageLayout() != VK_IMAGE_LAYOUT_GENERAL;
      VkPipelineStageFlags2 sourceStages = 0;
      VkAccessFlags2 sourceAccess = 0;
      // Dependencies on this queue become barrier scopes; other queues require timeline waits.
      auto depend = [&](const Access& prior) {
        if (!prior.stageMask) return;
        if (prior.queue == queue) {
          sourceStages |= prior.stageMask;
          sourceAccess |= prior.accessMask;
        } else {
          _addWait(waits[pass], _queueTimelines.at(prior.queue).semaphore->getSemaphore(), prior.value,
                   access.stageMask);
        }
      };
      // Every read depends on lastWrite; writes and layout transitions also depend on all readers.
      depend(history.lastWrite);
      if (access.writes || transition) {
        for (const auto& reader : history.readers) depend(reader);
      }
      if (access.buffer && sourceStages) {
        barrier.addBuffer(access.buffer, sourceStages, sourceAccess, access.stageMask, access.accessMask);
      } else if (access.image && (sourceStages || transition)) {
        auto& image = *access.image;
        barrier.addImage(image.getImage(), sourceStages ? sourceStages : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         sourceAccess, transition ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : access.stageMask,
                         transition ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT : access.accessMask,
                         transition ? image.getImageLayout() : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                          static_cast<uint32_t>(image.getLayerNumber())});
      }
      if (transition) {
        access.image->setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
        history.lastWrite = {queue, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, value};
        history.readers.clear();
      }
      // A write starts new history; a read preserves lastWrite and joins this queue's readers.
      const Access current{queue, access.stageMask, access.accessMask, value};
      if (access.writes) {
        history.lastWrite = current;
        history.readers.clear();
      } else {
        auto reader = std::ranges::find(history.readers, current.queue, &Access::queue);
        if (reader == history.readers.end()) {
          history.readers.push_back(current);
        } else {
          reader->stageMask |= current.stageMask;
          reader->accessMask |= current.accessMask;
          reader->value = std::max(reader->value, current.value);
        }
      }
    }
  }
}

bool Graph::render() {
  if (_passesOrdered.empty()) {
    return false;
  }

  _tracer->markFrame();
  const auto cpuTrace = _tracer->traceCPU("Graph.render");
  // timeline semaphore instead of fence
  if (_frameNumber >= _maxFramesInFlight) {
    const auto waitCpuTrace = _tracer->traceCPU("Graph.waitForFrame");
    uint64_t waitValue = _frameNumber - _maxFramesInFlight + 1;
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

  uint32_t swapchainIndex = 0;
  if (_swapchain != nullptr) {
    auto status = _swapchain->acquireNextImage(*_semaphoreImageAvailable[getFrameInFlight()]);
    if (status == VK_ERROR_OUT_OF_DATE_KHR) {
      return true;
    }
    if (status != VK_SUCCESS && status != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to acquire swap chain image");
    }
    swapchainIndex = _swapchain->getSwapchainIndex();
  }

  // Resolve physical accesses and prepare their dependencies before launching parallel command recording.
  std::unordered_map<GraphPass*, uint64_t> submissionValues;
  std::unordered_map<GraphPass*, std::vector<VkSemaphoreSubmitInfo>> physicalWaits;
  _preparePhysicalAccesses(submissionValues, physicalWaits);

  // Record all pass command buffers.
  const Sync emptySync;
  std::vector<std::future<void>> futureTasks =
      _passesOrdered | std::views::transform([this, &emptySync, swapchainIndex](GraphPass* pass) {
        auto commandBuffer = pass->getCommandBuffers()[getFrameInFlight()];
        if (!commandBuffer->getActive()) {
          commandBuffer->beginCommands();
        }

        if (pass == _swapchainFirstPass) {
          auto& image = _swapchain->getImage(swapchainIndex);
          Barrier acquiredImage;
          acquiredImage.addImage(image.getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, _swapchainFirstStageMask,
                                 _swapchainFirstAccessMask, image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
                                 {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                                  static_cast<uint32_t>(image.getLayerNumber())});
          acquiredImage.execute(commandBuffer->getCommandBuffer());
          image.setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
        }

        if (const auto initialIt = _initialImageBarriers.find(pass); initialIt != _initialImageBarriers.end()) {
          // An upload may have already performed this one-time transition after calculate().
          for (const auto& [image, barrier] : initialIt->second) {
            if (image->getImageLayout() == VK_IMAGE_LAYOUT_GENERAL) continue;
            barrier.execute(commandBuffer->getCommandBuffer());
            image->setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
          }
          _initialImageBarriers.erase(initialIt);
        }

        const QueueType queueType = _passQueueTypes.at(pass);

        if (_collectionPasses.contains(pass)) {
          _tracer->collect(queueType, *commandBuffer);
        }

        const auto syncIt = _sync.find(pass);
        const auto& sync = syncIt != _sync.end() ? syncIt->second : emptySync;
        return _threadPool->submit([this, pass, commandBuffer, queueType, &sync]() {
          const auto passCpuTrace = _tracer->traceCPU("Graph.pass." + pass->getName());
          const auto passGpuTrace = _tracer->traceGPU("Graph.pass." + pass->getName(), queueType, *commandBuffer);
          for (const Barrier* barrier : sync.getBarriers()) {
            barrier->execute(commandBuffer->getCommandBuffer());
          }
          if (const auto physical = _physicalBarriers.find(pass); physical != _physicalBarriers.end()) {
            physical->second.execute(commandBuffer->getCommandBuffer());
          }
          pass->execute(getFrameInFlight(), *commandBuffer);
        });
      }) |
      std::ranges::to<std::vector<std::future<void>>>();

  try {
    for (auto& task : futureTasks) task.get();
  } catch (...) {
    for (auto& task : futureTasks) {
      if (task.valid()) task.wait();
    }
    throw;
  }
  auto submitPassToQueue = [this, &submissionValues](
                               GraphPass* previousPass, const std::vector<CommandBuffer*>& commandBufferSubmit,
                               const std::vector<VkSemaphoreSubmitInfo>& waitSemaphoreInfos,
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

    std::vector<VkSemaphoreSubmitInfo> signalSemaphoreInfos;
    signalSemaphoreInfos.reserve(signalSemaphores.size());
    for (std::size_t i = 0; i < signalSemaphores.size(); ++i) {
      signalSemaphoreInfos.push_back({
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = signalSemaphores[i],
          .value = signalValues ? signalValues->at(i) : 0,
          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      });
    }

    const QueueType queueType = _passQueueTypes.at(previousPass);
    if (const auto timeline = _queueTimelines.find(_device->getQueue(queueType)); timeline != _queueTimelines.end()) {
      signalSemaphoreInfos.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                      .semaphore = timeline->second.semaphore->getSemaphore(),
                                      .value = submissionValues.at(previousPass),
                                      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
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
  std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
  for (const auto& ticket : _pendingTickets) {
    waitSemaphores.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                              .semaphore = ticket.semaphore,
                              .value = ticket.value,
                              .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
  }
  GraphPass* previousPass = nullptr;
  for (auto* pass : _passesOrdered) {
    if (previousPass != nullptr) {
      const bool queueTypeChange = _passQueueTypes.at(previousPass) != _passQueueTypes.at(pass);
      if (queueTypeChange) {
        submitPassToQueue(previousPass, commandBufferSubmit, waitSemaphores, signalSemaphores);
        commandBufferSubmit.clear();
        waitSemaphores.clear();
        signalSemaphores.clear();
      }
    }

    previousPass = pass;
    commandBufferSubmit.push_back(pass->getCommandBuffers()[getFrameInFlight()]);
    if (const auto syncIt = _sync.find(pass); syncIt != _sync.end()) {
      for (Semaphore* semaphore : syncIt->second.getWaitSemaphores()) {
        waitSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = semaphore->getSemaphore(),
            .value = 0,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        });
      }
      for (Semaphore* semaphore : syncIt->second.getSignalSemaphores()) {
        signalSemaphores.push_back(semaphore->getSemaphore());
      }
    }
    if (const auto physical = physicalWaits.find(pass); physical != physicalWaits.end()) {
      for (const auto& wait : physical->second) {
        _addWait(waitSemaphores, wait.semaphore, wait.value, wait.stageMask);
      }
    }
  }

  if (_swapchain != nullptr) {
    const auto transitionGpuTrace = _tracer->traceGPU(
        "Graph.transition.present", _passQueueTypes.at(_passesOrdered.back()), *commandBufferSubmit.back());
    auto& image = _swapchain->getImage(swapchainIndex);
    Barrier present;
    present.addImage(image.getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                      static_cast<uint32_t>(image.getLayerNumber())});
    present.execute(commandBufferSubmit.back()->getCommandBuffer());
    image.setImageLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if (auto history = _imageHistory.find(image.getImage()); history != _imageHistory.end()) {
      history->second.lastWrite = {_device->getQueue(_passQueueTypes.at(previousPass)),
                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   submissionValues.at(previousPass)};
      history->second.readers.clear();
    }
  }

  std::vector<uint64_t> signalValues(signalSemaphores.size() + 1, 0);
  signalValues[signalSemaphores.size()] = _frameNumber + 1;
  signalSemaphores.push_back(_semaphoreInFlight->getSemaphore());

  submitPassToQueue(_passesOrdered.back(), commandBufferSubmit, waitSemaphores, signalSemaphores, signalValues);
  _pendingTickets.clear();

  ++_frameNumber;

  if (_swapchain != nullptr) {
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
    auto result = vkQueuePresentKHR(_device->getQueue(QueueType::GENERAL), &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to present swap chain image: " + std::to_string(result));
    }
  }

  return false;
}

void Graph::reset(glm::ivec2 resolution) {
  const auto cpuTrace = _tracer->traceCPU("Graph.reset");
  if (_swapchain == nullptr) {
    throw std::logic_error("Can't reset presentation without a swapchain");
  }

  // wait all queues idle before reset
  if (vkDeviceWaitIdle(_device->getLogicalDevice()) != VK_SUCCESS) throw std::runtime_error("failed to reset");

  if (resolution.x == 0 || resolution.y == 0) throw std::runtime_error("Can't reset if resolution is 0");

  auto oldSwapchain = _swapchain->reset(resolution);
  _graphStorage->reset(oldSwapchain, _swapchain->getImageViews());

  _semaphoreRenderFinished.clear();
  std::ranges::generate_n(std::back_inserter(_semaphoreRenderFinished), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });

  for (auto&& pass : _passesOrdered) {
    pass->reset(_swapchain->getImageViews());
  }

  calculate();
}
