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

CommandBuffer& GraphPass::getCommandBuffer(int frame) const { return *_commandBuffers.at(frame); }

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
  it->stageMask |= resource.stageMask;
  it->accessMask |= resource.accessMask;
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

  for (auto* value : _passesOrdered) {
    std::cout << "========================================" << std::endl;
    std::cout << "Name: " << value->getName()
              << ", Stage: " << (value->getGraphPassType() == GraphPassType::GRAPHIC ? "GRAPHIC" : "COMPUTE")
              << std::endl;
    std::cout << " queue: " << (value->getQueueType() == QueueType::COMPUTE ? "COMPUTE" : "GENERAL") << std::endl;
    std::cout << " barriers: ";
    const auto barrier = _physicalBarriers.find(value);
    if (barrier != _physicalBarriers.end() &&
        (!barrier->second.getBufferBarriers().empty() || !barrier->second.getImageBarriers().empty())) {
      std::cout << std::endl;
      printBarrier(&barrier->second, 0);
    } else {
      std::cout << "<none>" << std::endl;
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
  _physicalBarriers.clear();
  _resolvedAccesses.clear();
  _submissions.clear();
  _swapchainFirstPass = nullptr;

  if (_passes.empty()) {
    return;
  }

  auto addResources = [this](GraphPass* pass, const auto& names, Resource::Type type, Resource::Operation operation,
                             VkPipelineStageFlags2 stages, VkAccessFlags2 access) {
    for (const auto& name : names) {
      _resources[pass].add({.name = name,
                            .type = type,
                            .operation = static_cast<uint8_t>(operation),
                            .stageMask = stages,
                            .accessMask = access});
    }
  };

  for (auto&& pass : _passes) {
    _resources.try_emplace(pass.get());
    if (pass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* compute = static_cast<GraphPassCompute*>(pass.get());
      addResources(pass.get(), compute->getStorageBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      addResources(pass.get(), compute->getIndirectBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
      addResources(pass.get(), compute->getStorageTextureInputs(), Resource::Type::IMAGE, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      addResources(pass.get(), compute->getStorageBufferOutputs(), Resource::Type::BUFFER, Resource::Operation::WRITE,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
      addResources(pass.get(), compute->getStorageTextureOutputs(), Resource::Type::IMAGE, Resource::Operation::WRITE,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    } else if (pass->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto* graphic = static_cast<GraphPassGraphic*>(pass.get());
      addResources(pass.get(), graphic->getVertexBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
      addResources(pass.get(), graphic->getIndexBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT);
      addResources(pass.get(), graphic->getStorageBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      addResources(pass.get(), graphic->getIndirectBufferInputs(), Resource::Type::BUFFER, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
      addResources(pass.get(), graphic->getTextureInputs(), Resource::Type::IMAGE, Resource::Operation::READ,
                   VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_SHADER_READ_BIT);

      auto addTarget = [&](const auto& name, VkPipelineStageFlags2 stages, VkAccessFlags2 read, VkAccessFlags2 write) {
        uint8_t operation = static_cast<uint8_t>(Resource::Operation::WRITE);
        // KEEP loads the previous attachment contents, so the target is both read and written.
        if (graphic->getTargetLoadAction(name) == RenderTargetLoadAction::KEEP) {
          operation |= static_cast<uint8_t>(Resource::Operation::READ);
          write |= read;
        }
        addResources(pass.get(), std::views::single(name), Resource::Type::IMAGE,
                     static_cast<Resource::Operation>(operation), stages, write);
      };
      for (const auto& colorTarget : graphic->getColorTargets()) {
        addTarget(colorTarget, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      }
      if (const auto& depth = graphic->getDepthTarget()) {
        addTarget(*depth, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
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

  // Prepare every used handler and keep history for physical instances still in the graph.
  std::unordered_set<VkBuffer> buffers;
  std::unordered_set<VkImage> images;
  std::unordered_set<VkQueue> collectionQueues;
  for (std::size_t passIndex = 0; passIndex < _passesOrdered.size(); ++passIndex) {
    auto* pass = _passesOrdered[passIndex];
    const QueueType queueType = pass->getQueueType();
    const auto queue = _device->getQueue(queueType);
    _passQueueTypes.emplace(pass, queueType);
    if (collectionQueues.insert(queue).second) {
      _collectionPasses.insert(pass);
    }
    if (_submissions.empty() || _passQueueTypes.at(_passesOrdered[passIndex - 1]) != queueType) {
      _submissions.push_back({.queue = queue, .firstPass = passIndex});
    }
    _submissions.back().endPass = passIndex + 1;
    _physicalBarriers.try_emplace(pass);
    for (const auto& resource : _resources.at(pass).getResources()) {
      if (resource.type == Resource::Type::BUFFER) {
        for (const auto* buffer : _graphStorage->getBufferHandler(resource.name).getBuffers()) {
          buffers.insert(buffer->getBuffer());
          _bufferHistory.try_emplace(buffer->getBuffer());
        }
      } else {
        for (const auto* view : _graphStorage->getImageViewHolder(resource.name).getImageViews()) {
          images.insert(view->getImage().getImage());
          _imageHistory.try_emplace(view->getImage().getImage());
        }
      }
    }
    _resolvedAccesses.reserve(_resources.at(pass).getResources().size());
  }
  std::erase_if(_bufferHistory, [&](const auto& entry) { return !buffers.contains(entry.first); });
  std::erase_if(_imageHistory, [&](const auto& entry) { return !images.contains(entry.first); });

  // Multiple real queues use timelines for resource dependencies and completion of the whole frame.
  if (collectionQueues.size() > 1) {
    for (const auto queue : collectionQueues) {
      auto [timeline, added] = _queueTimelines.try_emplace(queue);
      if (added) timeline->second.semaphore = std::make_unique<Semaphore>(VK_SEMAPHORE_TYPE_TIMELINE, *_device);
    }
  }
  for (auto& [buffer, history] : _bufferHistory) history.reserve(collectionQueues.size() + 1);
  for (auto& [image, history] : _imageHistory) history.reserve(collectionQueues.size() + 1);

  for (auto& submission : _submissions) {
    submission.commands.resize(_maxFramesInFlight);
    for (int frame = 0; frame < _maxFramesInFlight; ++frame) {
      auto& commands = submission.commands[frame];
      commands.reserve(submission.endPass - submission.firstPass);
      for (auto index = submission.firstPass; index < submission.endPass; ++index) {
        commands.push_back({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                            .commandBuffer = _passesOrdered[index]->getCommandBuffer(frame).getCommandBuffer()});
      }
    }
    submission.waits.reserve(_queueTimelines.size() + _pendingTickets.size() + (_swapchain ? 1 : 0));
    if (const auto timeline = _queueTimelines.find(submission.queue); timeline != _queueTimelines.end()) {
      submission.signals.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                    .semaphore = timeline->second.semaphore->getSemaphore(),
                                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
    }
  }
  auto& signals = _submissions.back().signals;
  if (_swapchain) {
    signals.push_back(
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
  }
  signals.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                     .semaphore = _semaphoreInFlight->getSemaphore(),
                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
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

void Graph::_preparePhysicalAccesses() {
  _swapchainFirstPass = nullptr;
  // Assign one completion value to each prepared submission.
  for (auto& submission : _submissions) {
    submission.waits.clear();
    if (const auto timeline = _queueTimelines.find(submission.queue); timeline != _queueTimelines.end()) {
      submission.value = ++timeline->second.value;
      submission.signals.front().value = submission.value;
    }
  }
  const VkImage acquiredImage = _swapchain ? _swapchain->getImage(_swapchain->getSwapchainIndex()).getImage()
                                           : VK_NULL_HANDLE;
  for (auto& submission : _submissions) {
    const auto queue = submission.queue;
    const auto value = submission.value;
    for (auto index = submission.firstPass; index < submission.endPass; ++index) {
      auto* pass = _passesOrdered[index];
      auto& barrier = _physicalBarriers.at(pass);
      barrier.clear();
      _resolvedAccesses.clear();
      // Combine aliases into one READ/WRITE history update per selected instance in this pass.
      for (const auto& resource : _resources.at(pass).getResources()) {
        Access current{.queue = queue,
                       .stageMask = resource.stageMask,
                       .accessMask = resource.accessMask,
                       .value = value,
                       .writes = (resource.operation & static_cast<uint8_t>(Resource::Operation::WRITE)) != 0};
        if (resource.type == Resource::Type::BUFFER) {
          current.buffer = &_graphStorage->getBufferHandler(resource.name).getBuffer();
        } else {
          current.image = &_graphStorage->getImageViewHolder(resource.name).getImageView().getImage();
        }
        auto existing = std::ranges::find_if(_resolvedAccesses, [&](const Access& other) {
          return current.buffer ? other.buffer && other.buffer->getBuffer() == current.buffer->getBuffer()
                                : other.image && other.image->getImage() == current.image->getImage();
        });
        if (existing == _resolvedAccesses.end()) {
          _resolvedAccesses.push_back(current);
        } else {
          existing->stageMask |= current.stageMask;
          existing->accessMask |= current.accessMask;
          existing->writes |= current.writes;
        }
      }
      for (const auto& access : _resolvedAccesses) {
        auto& history = access.buffer ? _bufferHistory.at(access.buffer->getBuffer())
                                      : _imageHistory.at(access.image->getImage());
        if (access.image && access.image->getImage() == acquiredImage && !_swapchainFirstPass) {
          _swapchainFirstPass = pass;
          _addWait(submission.waits, _semaphoreImageAvailable[getFrameInFlight()]->getSemaphore(), 0,
                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }
        const bool transition = access.image && access.image->getImageLayout() != VK_IMAGE_LAYOUT_GENERAL;
        VkPipelineStageFlags2 sourceStages = 0;
        VkAccessFlags2 sourceAccess = 0;
        // Reads depend on writes; writes and layout transitions depend on every prior access.
        for (const auto& prior : history) {
          if (!prior.writes && !access.writes && !transition) continue;
          if (prior.queue == queue) {
            sourceStages |= prior.stageMask;
            sourceAccess |= prior.accessMask;
          } else {
            _addWait(submission.waits, _queueTimelines.at(prior.queue).semaphore->getSemaphore(), prior.value,
                     transition ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : access.stageMask);
          }
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
          history.assign(1, access);
          history.front().stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
          history.front().accessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
          history.front().writes = true;
        }
        // A WRITE replaces history; a READ keeps the writer and joins this queue's readers.
        if (access.writes) {
          history.assign(1, access);
        } else {
          auto reader = std::ranges::find_if(
              history, [&](const Access& prior) { return !prior.writes && prior.queue == queue; });
          if (reader == history.end()) {
            history.push_back(access);
          } else {
            reader->stageMask |= access.stageMask;
            reader->accessMask |= access.accessMask;
            reader->value = std::max(reader->value, value);
          }
        }
      }
    }
    // Upload tickets have no per-resource metadata, so every submission must wait for them.
    for (const auto& ticket : _pendingTickets) {
      _addWait(submission.waits, ticket.semaphore, ticket.value, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }
  }
  auto& last = _submissions.back();
  if (_swapchain) {
    if (!_swapchainFirstPass) {
      // Even an unused acquired image must be transitioned before presentation.
      _swapchainFirstPass = _passesOrdered.back();
      auto& image = _swapchain->getImage(_swapchain->getSwapchainIndex());
      _physicalBarriers.at(_swapchainFirstPass)
          .addImage(image.getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, image.getImageLayout(),
                    VK_IMAGE_LAYOUT_GENERAL,
                    {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                     static_cast<uint32_t>(image.getLayerNumber())});
      image.setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
      _addWait(last.waits, _semaphoreImageAvailable[getFrameInFlight()]->getSemaphore(), 0,
               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }
    last.signals[last.signals.size() - 2].semaphore =
        _semaphoreRenderFinished[_swapchain->getSwapchainIndex()]->getSemaphore();
  }
  // Frame-slot reuse and presentation must wait for every queue, including independent passes.
  for (const auto& [queue, timeline] : _queueTimelines) {
    if (queue != last.queue) {
      _addWait(last.waits, timeline.semaphore->getSemaphore(), timeline.value, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }
  }
  last.signals.back().value = _frameNumber + 1;
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
  const int frame = getFrameInFlight();
  _preparePhysicalAccesses();

  std::vector<std::future<void>> futureTasks =
      _passesOrdered | std::views::transform([this, frame](GraphPass* pass) {
        auto* commandBuffer = &pass->getCommandBuffer(frame);
        if (!commandBuffer->getActive()) commandBuffer->beginCommands();
        const QueueType queueType = _passQueueTypes.at(pass);
        if (_collectionPasses.contains(pass)) _tracer->collect(queueType, *commandBuffer);
        return _threadPool->submit([this, pass, commandBuffer, queueType, frame]() {
          const auto passCpuTrace = _tracer->traceCPU("Graph.pass." + pass->getName());
          const auto passGpuTrace = _tracer->traceGPU("Graph.pass." + pass->getName(), queueType, *commandBuffer);
          _physicalBarriers.at(pass).execute(commandBuffer->getCommandBuffer());
          pass->execute(frame, *commandBuffer);
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

  if (_swapchain != nullptr) {
    auto* commandBuffer = &_passesOrdered.back()->getCommandBuffer(frame);
    const auto transitionGpuTrace = _tracer->traceGPU("Graph.transition.present",
                                                      _passQueueTypes.at(_passesOrdered.back()), *commandBuffer);
    auto& image = _swapchain->getImage(swapchainIndex);
    Barrier present;
    present.addImage(image.getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                      static_cast<uint32_t>(image.getLayerNumber())});
    present.execute(commandBuffer->getCommandBuffer());
    image.setImageLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if (auto history = _imageHistory.find(image.getImage()); history != _imageHistory.end()) {
      history->second.assign(1, {.queue = _submissions.back().queue,
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .accessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .value = _submissions.back().value,
                                 .image = &image,
                                 .writes = true});
    }
  }

  for (auto& submission : _submissions) {
    for (auto index = submission.firstPass; index < submission.endPass; ++index) {
      _passesOrdered[index]->getCommandBuffer(frame).endCommands();
    }
    const auto& commands = submission.commands[frame];
    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(submission.waits.size()),
        .pWaitSemaphoreInfos = submission.waits.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(commands.size()),
        .pCommandBufferInfos = commands.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(submission.signals.size()),
        .pSignalSemaphoreInfos = submission.signals.data(),
    };
    const auto result = vkQueueSubmit2(submission.queue, 1, &submitInfo, nullptr);
    if (result != VK_SUCCESS) throw std::runtime_error("vkQueueSubmit2 failed: " + std::to_string(result));
  }
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
