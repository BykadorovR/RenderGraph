module;

#include "BS_thread_pool.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
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

// Access bits emitted by graph declarations and its layout transitions.
bool Graph::_hasReadAccess(VkAccessFlags2 accessMask) const noexcept {
  constexpr VkAccessFlags2 readAccessBits = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                                            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
  return (accessMask & readAccessBits) != 0;
}

bool Graph::_hasWriteAccess(VkAccessFlags2 accessMask) const noexcept {
  constexpr VkAccessFlags2 writeAccessBits = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                             VK_ACCESS_2_MEMORY_WRITE_BIT;
  return (accessMask & writeAccessBits) != 0;
}

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
  for (auto&& [name, resource] : _images) {
    if (resource.imageViewHolder->contains(oldSwapchain)) {
      resource.imageViewHolder->setImageViews(newSwapchain);
      continue;
    }
    if (resource.resizePolicy == ImageResizePolicy::FIXED) {
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
  for (const auto& passResourcesEntry : _resourcesByPass) {
    for (const Resource& declaredResource : passResourcesEntry.second) {
      if (declaredResource.type == Resource::Type::BUFFER) {
        const auto bufferInstances = _graphStorage->getBufferHandler(declaredResource.name).getBuffers();
        for (std::size_t instanceIndex = 0; instanceIndex < bufferInstances.size(); ++instanceIndex) {
          _tracer->setObjectName(VK_OBJECT_TYPE_BUFFER,
                                 std::bit_cast<uint64_t>(bufferInstances[instanceIndex]->getBuffer()),
                                 declaredResource.name + ".buffer[" + std::to_string(instanceIndex) + "]");
        }
      } else {
        const auto imageViewInstances = _graphStorage->getImageViewHolder(declaredResource.name).getImageViews();
        for (std::size_t instanceIndex = 0; instanceIndex < imageViewInstances.size(); ++instanceIndex) {
          const std::string suffix = "[" + std::to_string(instanceIndex) + "]";
          _tracer->setObjectName(VK_OBJECT_TYPE_IMAGE,
                                 std::bit_cast<uint64_t>(imageViewInstances[instanceIndex]->getImage().getImage()),
                                 declaredResource.name + ".image" + suffix);
          _tracer->setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW,
                                 std::bit_cast<uint64_t>(imageViewInstances[instanceIndex]->getImageView()),
                                 declaredResource.name + ".imageView" + suffix);
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
  std::ranges::generate_n(std::back_inserter(_swapchainImageAvailableSemaphores), _maxFramesInFlight,
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
  std::ranges::generate_n(std::back_inserter(_swapchainRenderFinishedSemaphores), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });
}

void Graph::initialize() { _frameSlotCompletions.resize(_maxFramesInFlight); }

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

void Graph::print() const {
  if (_resourcesByPass.empty()) return;

  const bool synchronizationPrepared = !_frameSubmissions.empty() && _frameNumber > 0;
  if (synchronizationPrepared) {
    const auto preparedFrameNumber = _frameNumber - 1;
    std::cout << "Synchronization: prepared frame " << preparedFrameNumber << " (slot "
              << preparedFrameNumber % _maxFramesInFlight << ")" << std::endl;
  } else {
    std::cout << "Synchronization: not prepared; submissions, barriers and waits are resolved by render()" << std::endl;
  }

  auto findBufferName = [this](VkBuffer targetBuffer) -> std::string {
    for (const auto& [graphPass, passResources] : _resourcesByPass) {
      for (const Resource& declaredResource : passResources) {
        if (declaredResource.type != Resource::Type::BUFFER) {
          continue;
        }
        for (const auto* buffer : _graphStorage->getBufferHandler(declaredResource.name).getBuffers()) {
          if (buffer->getBuffer() == targetBuffer) {
            return declaredResource.name;
          }
        }
      }
    }

    return "<unknown>";
  };

  auto findImageName = [this](VkImage targetImage) -> std::string {
    for (const auto& [graphPass, passResources] : _resourcesByPass) {
      for (const Resource& declaredResource : passResources) {
        if (declaredResource.type != Resource::Type::IMAGE) {
          continue;
        }
        const auto& imageViews = _graphStorage->getImageViewHolder(declaredResource.name).getImageViews();
        for (const auto& imageView : imageViews) {
          if (imageView->getImage().getImage() == targetImage) {
            return declaredResource.name;
          }
        }
      }
    }

    return "<unknown>";
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

  auto printSemaphores = [](std::string_view label, const auto& semaphoreOperations) {
    std::cout << label;
    if (semaphoreOperations.empty()) std::cout << " <none>";
    std::cout << std::endl;
    for (const auto& semaphoreOperation : semaphoreOperations) {
      std::cout << "  " << semaphoreOperation.semaphore << " value=" << semaphoreOperation.value << " stages=0x"
                << std::hex << semaphoreOperation.stageMask << std::dec << std::endl;
    }
  };

  auto printBarrier = [&](const Barrier& passBarriers) {
    for (const auto& bufferBarrier : passBarriers.getBufferBarriers()) {
      std::cout << "  buffer " << findBufferName(bufferBarrier.buffer) << " (" << bufferBarrier.buffer << ") stages 0x"
                << std::hex << bufferBarrier.srcStageMask << " -> 0x" << bufferBarrier.dstStageMask << ", access 0x"
                << bufferBarrier.srcAccessMask << " -> 0x" << bufferBarrier.dstAccessMask << std::dec << std::endl;
    }
    for (const auto& imageBarrier : passBarriers.getImageBarriers()) {
      std::cout << "  image " << findImageName(imageBarrier.image) << " (" << imageBarrier.image << ") stages 0x"
                << std::hex << imageBarrier.srcStageMask << " -> 0x" << imageBarrier.dstStageMask << ", access 0x"
                << imageBarrier.srcAccessMask << " -> 0x" << imageBarrier.dstAccessMask << std::dec;
      if (imageBarrier.oldLayout != imageBarrier.newLayout) {
        std::cout << ", layout " << static_cast<int>(imageBarrier.oldLayout) << " -> "
                  << static_cast<int>(imageBarrier.newLayout);
      }
      std::cout << std::endl;
    }
  };

  for (std::size_t submissionIndex = 0; submissionIndex < _frameSubmissions.size(); ++submissionIndex) {
    const auto& frameSubmission = _frameSubmissions[submissionIndex];
    std::cout << "Submission " << submissionIndex << ", queue: " << frameSubmission.queue << ", passes:";
    for (auto passIndex = frameSubmission.firstPassIndex; passIndex < frameSubmission.endPassIndex; ++passIndex) {
      std::cout << " " << _passes[passIndex]->getName();
    }
    std::cout << std::endl;
    if (synchronizationPrepared) {
      printSemaphores(" waits:", frameSubmission.semaphoreWaits);
      printSemaphores(" signals:", frameSubmission.semaphoreSignals);
    }
  }

  for (const auto& graphPass : _passes) {
    std::cout << "========================================" << std::endl;
    std::cout << "Name: " << graphPass->getName()
              << ", Stage: " << (graphPass->getGraphPassType() == GraphPassType::GRAPHIC ? "GRAPHIC" : "COMPUTE")
              << std::endl;
    std::cout << " queue: " << (graphPass->getQueueType() == QueueType::COMPUTE ? "COMPUTE" : "GENERAL") << std::endl;
    for (const auto& declaredResource : _resourcesByPass.at(graphPass.get())) {
      const bool resourceReads = _hasReadAccess(declaredResource.accessMask);
      const bool resourceWrites = _hasWriteAccess(declaredResource.accessMask);
      std::cout << " resource: " << declaredResource.name << " ("
                << (declaredResource.type == Resource::Type::BUFFER ? "buffer" : "image") << ") "
                << (resourceReads ? (resourceWrites ? "READ|WRITE" : "READ") : "WRITE") << " stages=0x" << std::hex
                << declaredResource.stageMask << " access=0x" << declaredResource.accessMask << std::dec;
      if (declaredResource.type == Resource::Type::BUFFER) {
        printRange(" handles: ", _graphStorage->getBufferHandler(declaredResource.name).getBuffers(),
                   [](const auto* buffer) { return buffer->getBuffer(); });
      } else {
        printRange(" handles: ", _graphStorage->getImageViewHolder(declaredResource.name).getImageViews(),
                   [](const auto* view) { return view->getImage().getImage(); });
      }
    }
    std::cout << " barriers: ";
    const auto passBarriers = _barriersBeforePass.find(graphPass.get());
    if (!synchronizationPrepared) {
      std::cout << "<resolved by render()>" << std::endl;
    } else if (passBarriers != _barriersBeforePass.end() && (!passBarriers->second.getBufferBarriers().empty() ||
                                                             !passBarriers->second.getImageBarriers().empty())) {
      std::cout << std::endl;
      printBarrier(passBarriers->second);
    } else {
      std::cout << "<none>" << std::endl;
    }

    printRange(" command buffers: ", graphPass->getCommandBuffers(),
               [](const auto* commandBuffer) { return commandBuffer->getCommandBuffer(); });
  }

  std::cout << "========================================" << std::endl;
}

void Graph::addFrameWait(GpuTicket ticket) {
  const std::scoped_lock lock(_pendingWaitsMutex);
  _pendingWaits.push_back({.completion = ticket});
}

void Graph::addResourceWait(Buffer& buffer, GpuTicket ticket) {
  const std::scoped_lock lock(_pendingWaitsMutex);
  _pendingWaits.push_back({.accessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .completion = ticket, .buffer = &buffer});
}

void Graph::addResourceWait(Image& image, GpuTicket ticket) {
  const std::scoped_lock lock(_pendingWaitsMutex);
  _pendingWaits.push_back({.accessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .completion = ticket, .image = &image});
}

void Graph::_applyWait(Access externalWrite, std::vector<GpuTicket>& frameWaits) {
  if (!externalWrite.buffer && !externalWrite.image) {
    const auto ticket = externalWrite.completion;
    const auto existingFrameWait = std::ranges::find(frameWaits, ticket.semaphore, &GpuTicket::semaphore);
    if (existingFrameWait == frameWaits.end())
      frameWaits.push_back(ticket);
    else
      existingFrameWait->value = std::max(existingFrameWait->value, ticket.value);
    return;
  }
  auto& resource = externalWrite.buffer ? _bufferHistory[externalWrite.buffer->getBuffer()]
                                        : _imageHistory[externalWrite.image->getImage()];
  auto& history = resource.accesses;
  // New uploads replace previous GPU accesses; registrations within one frame accumulate.
  if (resource.uploadBatch != _uploadBatch) {
    history.clear();
    resource.uploadBatch = _uploadBatch;
  }
  auto existing = std::ranges::find(history, externalWrite.completion.semaphore,
                                    [](const Access& access) { return access.completion.semaphore; });
  if (existing == history.end())
    history.push_back(externalWrite);
  else
    existing->completion.value = std::max(existing->completion.value, externalWrite.completion.value);
}

void Graph::calculate() {
  const auto cpuTrace = _tracer->traceCPU("Graph.calculate");
  // Rebuild graph dependencies while keeping external readiness in place.
  auto clearGraphAccesses = [](auto& histories) {
    for (auto& [handle, history] : histories) {
      std::erase_if(history.accesses, [](const Access& access) { return access.queue != VK_NULL_HANDLE; });
    }
    std::erase_if(histories, [](const auto& entry) { return entry.second.accesses.empty(); });
  };
  clearGraphAccesses(_bufferHistory);
  clearGraphAccesses(_imageHistory);
  // Keep dependencies on previously submitted graph work.
  for (const auto& frameCompletions : _frameSlotCompletions) {
    for (const auto& completion : frameCompletions) addFrameWait(completion);
  }
  _resourcesByPass.clear();
  _firstPassesPerQueue.clear();
  _barriersBeforePass.clear();
  _frameSubmissions.clear();

  if (_passes.empty()) {
    return;
  }

  // Prepare usage declarations only; each selected buffer/image is resolved later in render().
  auto addPassResources = [this](GraphPass* graphPass, const auto& resourceNames, Resource::Type resourceType,
                                 VkPipelineStageFlags2 stageMask, VkAccessFlags2 accessMask) {
    for (const auto& resourceName : resourceNames) {
      auto& resources = _resourcesByPass[graphPass];
      const auto existing = std::ranges::find(resources, resourceName, &Resource::name);
      if (existing == resources.end()) {
        resources.push_back(
            {.name = resourceName, .type = resourceType, .stageMask = stageMask, .accessMask = accessMask});
      } else {
        if (existing->type != resourceType)
          throw std::logic_error("Resource with the same name has a different type: " + resourceName);
        existing->stageMask |= stageMask;
        existing->accessMask |= accessMask;
      }
    }
  };

  for (auto&& graphPass : _passes) {
    _resourcesByPass.try_emplace(graphPass.get());
    if (graphPass->getGraphPassType() == GraphPassType::COMPUTE) {
      auto* computePass = static_cast<GraphPassCompute*>(graphPass.get());
      addPassResources(graphPass.get(), computePass->getStorageBufferInputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      addPassResources(graphPass.get(), computePass->getIndirectBufferInputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
      addPassResources(graphPass.get(), computePass->getStorageTextureInputs(), Resource::Type::IMAGE,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      addPassResources(graphPass.get(), computePass->getStorageBufferOutputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
      addPassResources(graphPass.get(), computePass->getStorageTextureOutputs(), Resource::Type::IMAGE,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    } else if (graphPass->getGraphPassType() == GraphPassType::GRAPHIC) {
      auto* graphicsPass = static_cast<GraphPassGraphic*>(graphPass.get());
      addPassResources(graphPass.get(), graphicsPass->getVertexBufferInputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
      addPassResources(graphPass.get(), graphicsPass->getIndexBufferInputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT);
      addPassResources(graphPass.get(), graphicsPass->getStorageBufferInputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      addPassResources(graphPass.get(), graphicsPass->getIndirectBufferInputs(), Resource::Type::BUFFER,
                       VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
      addPassResources(graphPass.get(), graphicsPass->getTextureInputs(), Resource::Type::IMAGE,
                       VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_SHADER_READ_BIT);

      auto addAttachmentResource = [&](const auto& resourceName, VkPipelineStageFlags2 stageMask,
                                       VkAccessFlags2 readAccessMask, VkAccessFlags2 attachmentAccessMask) {
        // KEEP loads the previous attachment contents, so the target is both read and written.
        if (graphicsPass->getTargetLoadAction(resourceName) == RenderTargetLoadAction::KEEP) {
          attachmentAccessMask |= readAccessMask;
        }
        addPassResources(graphPass.get(), std::views::single(resourceName), Resource::Type::IMAGE, stageMask,
                         attachmentAccessMask);
      };
      for (const auto& colorAttachmentName : graphicsPass->getColorTargets()) {
        addAttachmentResource(colorAttachmentName, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      }
      if (const auto& depthAttachmentName = graphicsPass->getDepthTarget()) {
        addAttachmentResource(
            *depthAttachmentName,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
      }
    }
  }

  // Debug labels enumerate all physical instances without evaluating their index callbacks.
  _setResourceNames();

  // Keep one timeline and one profiling/frame-wait entry per physical queue.
  // Timelines and counters survive recalculation.
  std::unordered_set<VkQueue> uniqueQueues;
  for (const auto& graphPass : _passes) {
    const auto graphQueue = _device->getQueue(graphPass->getQueueType());
    auto& pool = _barrierCommandPools[graphQueue];
    if (!pool) pool = std::make_unique<CommandPool>(graphPass->getQueueType(), *_device);
    auto& buffers = _barrierCommandBuffers[graphPass.get()];
    buffers.resize(_maxFramesInFlight);
    for (auto& commands : buffers) {
      if (!commands) commands = std::make_unique<CommandBuffer>(*pool, *_device);
    }
    if (!uniqueQueues.insert(graphQueue).second) continue;
    _firstPassesPerQueue.insert(graphPass.get());
    auto& timeline = _queueTimelines[graphQueue];
    if (!timeline.semaphore) timeline.semaphore = std::make_unique<Semaphore>(VK_SEMAPHORE_TYPE_TIMELINE, *_device);
  }
}

// Add a wait description for an existing semaphore; no semaphore is created here.
// S=5 at COMPUTE + S=8 at VERTEX -> one wait for S=8 at COMPUTE|VERTEX.
// These stages belong to the waiting submission, not the signaling queue.
void Graph::_addSemaphoreWait(std::vector<VkSemaphoreSubmitInfo>& semaphoreWaits,
                              VkSemaphore waitSemaphore,
                              uint64_t requiredValue,
                              VkPipelineStageFlags2 waitStages) {
  auto existingWait = std::ranges::find(semaphoreWaits, waitSemaphore, &VkSemaphoreSubmitInfo::semaphore);
  if (existingWait != semaphoreWaits.end()) {
    existingWait->value = std::max(existingWait->value, requiredValue);
    existingWait->stageMask |= waitStages;
  } else {
    semaphoreWaits.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                              .semaphore = waitSemaphore,
                              .value = requiredValue,
                              .stageMask = waitStages});
  }
}

bool Graph::_waitCoversCurrentStages(VkPipelineStageFlags2 waitStages,
                                     VkPipelineStageFlags2 currentStages) const noexcept {
  // ALL_GRAPHICS is a separate bit; expand it before comparing wait and current access stages.
  if (waitStages & VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT) {
    waitStages |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  return (waitStages & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) || (waitStages & currentStages) == currentStages;
}

void Graph::_preparePhysicalAccesses(std::span<std::future<void>> passRecordingTasks) {
  _frameSubmissions.clear();
  std::vector<Access> pendingWaits;
  std::vector<GpuTicket> frameWaits;
  // Physical accesses of the current pass, with aliases merged.
  std::vector<Access> currentPassAccesses;
  // External waits still needed by this pass; may require a new submission.
  std::vector<VkSemaphoreSubmitInfo> currentPassExternalWaits;
  // Add the acquire semaphore wait at the first use of the current swapchain image.
  bool swapchainImageUsed = false;
  const VkImage swapchainImage = _swapchain ? _swapchain->getImage(_swapchain->getSwapchainIndex()).getImage()
                                            : VK_NULL_HANDLE;
  for (std::size_t currentPassIndex = 0; currentPassIndex < _passes.size(); ++currentPassIndex) {
    if (!passRecordingTasks.empty()) passRecordingTasks[currentPassIndex].get();
    {
      const std::scoped_lock lock(_pendingWaitsMutex);
      pendingWaits = _pendingWaits;
      _pendingWaits.clear();
    }
    for (const auto& wait : pendingWaits) _applyWait(wait, frameWaits);
    auto* currentPass = _passes[currentPassIndex].get();
    const auto currentQueue = _device->getQueue(currentPass->getQueueType());
    auto& currentPassBarriers = _barriersBeforePass[currentPass];
    currentPassBarriers.clear();
    currentPassAccesses.clear();
    currentPassAccesses.reserve(_resourcesByPass.at(currentPass).size());
    // Input=READ A and Output=WRITE A become one READ|WRITE access to A in this pass.
    for (const auto& declaredResource : _resourcesByPass.at(currentPass)) {
      Access currentAccess{.queue = currentQueue,
                           .stageMask = declaredResource.stageMask,
                           .accessMask = declaredResource.accessMask};
      if (declaredResource.type == Resource::Type::BUFFER) {
        currentAccess.buffer = &_graphStorage->getBufferHandler(declaredResource.name).getBuffer();
      } else {
        currentAccess.image = &_graphStorage->getImageViewHolder(declaredResource.name).getImageView().getImage();
      }
      auto existingPassAccess = std::ranges::find_if(currentPassAccesses, [&](const Access& passAccess) {
        return currentAccess.buffer
                   ? passAccess.buffer && passAccess.buffer->getBuffer() == currentAccess.buffer->getBuffer()
                   : passAccess.image && passAccess.image->getImage() == currentAccess.image->getImage();
      });
      if (existingPassAccess == currentPassAccesses.end())
        currentPassAccesses.push_back(currentAccess);
      else {
        existingPassAccess->stageMask |= currentAccess.stageMask;
        existingPassAccess->accessMask |= currentAccess.accessMask;
      }
    }

    currentPassExternalWaits.clear();
    for (const auto& access : currentPassAccesses) {
      const auto& history = access.buffer ? _bufferHistory[access.buffer->getBuffer()].accesses
                                          : _imageHistory[access.image->getImage()].accesses;
      if (history.empty() || history.front().queue != VK_NULL_HANDLE) continue;

      const auto stages = access.image && access.image->getImageLayout() != VK_IMAGE_LAYOUT_GENERAL
                              ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                              : access.stageMask;
      // A previous reader on this queue already waited for the external writers at its stages.
      VkPipelineStageFlags2 readerStages = 0;
      for (const auto& previous : history) {
        if (previous.queue == access.queue && !_hasWriteAccess(previous.accessMask)) readerStages |= previous.stageMask;
      }
      if (_waitCoversCurrentStages(readerStages, stages)) continue;

      for (const auto& previous : history) {
        if (previous.queue != VK_NULL_HANDLE) continue;
        const auto ticket = previous.completion;
        // A frame wait or a wait already scheduled on this queue may cover another resource from the same upload.
        VkPipelineStageFlags2 scheduledStages = 0;
        for (const auto& frameWait : frameWaits) {
          if (frameWait.semaphore == ticket.semaphore && frameWait.value >= ticket.value)
            scheduledStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        for (const auto& submission : _frameSubmissions) {
          if (_waitCoversCurrentStages(scheduledStages, stages)) break;
          if (submission.queue != access.queue) continue;
          for (const auto& wait : submission.semaphoreWaits) {
            if (wait.semaphore == ticket.semaphore && wait.value >= ticket.value) scheduledStages |= wait.stageMask;
          }
        }
        if (!_waitCoversCurrentStages(scheduledStages, stages))
          _addSemaphoreWait(currentPassExternalWaits, ticket.semaphore, ticket.value, stages);
      }
    }

    // Background -> Particles becomes [Background] [wait upload -> Particles].
    // Keep grouping when previous waits already cover the resource, value and destination stages.
    auto& currentQueueTimeline = _queueTimelines.at(currentQueue);
    if (_frameSubmissions.empty() || _frameSubmissions.back().queue != currentQueue ||
        !currentPassExternalWaits.empty()) {
      _frameSubmissions.push_back({.queue = currentQueue, .firstPassIndex = currentPassIndex});
      auto& currentSubmission = _frameSubmissions.back();
      // Assign completion values after splitting: two G submissions signal G=1 and G=2.
      currentSubmission.semaphoreSignals.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                                    .semaphore = currentQueueTimeline.semaphore->getSemaphore(),
                                                    .value = ++currentQueueTimeline.lastSignalValue,
                                                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
      for (const auto& externalWait : currentPassExternalWaits) {
        _addSemaphoreWait(currentSubmission.semaphoreWaits, externalWait.semaphore, externalWait.value,
                          externalWait.stageMask);
      }
    }
    auto& currentSubmission = _frameSubmissions.back();
    currentSubmission.endPassIndex = currentPassIndex + 1;

    const GpuTicket currentSubmissionCompletion{currentQueueTimeline.semaphore->getSemaphore(),
                                                currentQueueTimeline.lastSignalValue};
    for (auto currentPassAccess : currentPassAccesses) {
      currentPassAccess.completion = currentSubmissionCompletion;
      // Acquire signals readiness, not layout; wait before the first use of the swapchain image.
      if (currentPassAccess.image && currentPassAccess.image->getImage() == swapchainImage && !swapchainImageUsed) {
        swapchainImageUsed = true;
        _addSemaphoreWait(currentSubmission.semaphoreWaits,
                          _swapchainImageAvailableSemaphores[getFrameInFlight()]->getSemaphore(), 0,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      }
      auto access = currentPassAccess;
      auto& history = access.buffer ? _bufferHistory[access.buffer->getBuffer()].accesses
                                    : _imageHistory[access.image->getImage()].accesses;
      const bool transition = access.image && access.image->getImageLayout() != VK_IMAGE_LAYOUT_GENERAL;
      // A layout transition may write memory even for a read-only pass.
      if (transition) {
        access.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        access.accessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
      }
      const bool writes = _hasWriteAccess(access.accessMask);
      VkPipelineStageFlags2 sourceStages = 0;
      VkAccessFlags2 sourceAccess = 0;
      // External writers are covered by tickets. Graph accesses need a barrier or a queue wait.
      for (const auto& previous : history) {
        if (previous.queue == VK_NULL_HANDLE || (!_hasWriteAccess(previous.accessMask) && !writes)) continue;
        if (previous.queue == access.queue) {
          sourceStages |= previous.stageMask;
          sourceAccess |= previous.accessMask;
        } else {
          _addSemaphoreWait(currentSubmission.semaphoreWaits, previous.completion.semaphore, previous.completion.value,
                            access.stageMask);
        }
      }
      if (access.buffer && sourceStages) {
        currentPassBarriers.addBuffer(access.buffer, sourceStages, sourceAccess, access.stageMask, access.accessMask);
      } else if (access.image && (sourceStages || transition)) {
        auto& image = *access.image;
        currentPassBarriers.addImage(
            image.getImage(), sourceStages ? sourceStages : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, sourceAccess,
            access.stageMask, access.accessMask, image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
            {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
             static_cast<uint32_t>(image.getLayerNumber())});
        image.setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
      }
      // WRITE replaces history. A real write retains precise masks; a transition retains broad masks.
      if (writes) {
        history.assign(1, _hasWriteAccess(currentPassAccess.accessMask) ? currentPassAccess : access);
        continue;
      }
      // READ keeps the writer and merges this queue's readers.
      auto reader = std::ranges::find_if(history, [&](const Access& previous) {
        return !_hasWriteAccess(previous.accessMask) && previous.queue == access.queue;
      });
      if (reader == history.end()) {
        history.push_back(access);
      } else {
        reader->stageMask |= access.stageMask;
        reader->accessMask |= access.accessMask;
        reader->completion.value = std::max(reader->completion.value, access.completion.value);
      }
    }
  }
  ++_uploadBatch;
  // Later updates may register frame waits after a queue's first submission has been planned.
  for (auto& submission : _frameSubmissions) {
    if (!_firstPassesPerQueue.contains(_passes[submission.firstPassIndex].get())) continue;
    for (const auto& frameWait : frameWaits) {
      _addSemaphoreWait(submission.semaphoreWaits, frameWait.semaphore, frameWait.value,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }
  }
  if (!_swapchain) return;
  // Prepare swapchain acquire waits and presentation/completion signals.
  auto& lastSubmission = _frameSubmissions.back();
  // GENERAL -> PRESENT runs in the last command buffer and must follow every use of this image.
  // Wait only for its users on other queues, not for unrelated work submitted after them.
  if (const auto swapchainHistoryEntry = _imageHistory.find(swapchainImage);
      swapchainHistoryEntry != _imageHistory.end()) {
    for (const auto& previousAccess : swapchainHistoryEntry->second.accesses) {
      if (previousAccess.queue != lastSubmission.queue) {
        _addSemaphoreWait(lastSubmission.semaphoreWaits, previousAccess.completion.semaphore,
                          previousAccess.completion.value, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      }
    }
  }
  if (!swapchainImageUsed) {
    // No declared pass used the swapchain image, so the resource loop did not prepare it.
    // render() still records GENERAL -> PRESENT; prepare GENERAL and consume acquire here.
    auto& unusedSwapchainImage = _swapchain->getImage(_swapchain->getSwapchainIndex());
    _barriersBeforePass.at(_passes.back().get())
        .addImage(
            unusedSwapchainImage.getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            unusedSwapchainImage.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
            {unusedSwapchainImage.getAspectMask(), 0, static_cast<uint32_t>(unusedSwapchainImage.getMipMapNumber()), 0,
             static_cast<uint32_t>(unusedSwapchainImage.getLayerNumber())});
    unusedSwapchainImage.setImageLayout(VK_IMAGE_LAYOUT_GENERAL);
    _addSemaphoreWait(lastSubmission.semaphoreWaits,
                      _swapchainImageAvailableSemaphores[getFrameInFlight()]->getSemaphore(), 0,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  }
  // Present waits for this binary semaphore, which is separate from the acquire semaphore.
  lastSubmission.semaphoreSignals.push_back(
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = _swapchainRenderFinishedSemaphores[_swapchain->getSwapchainIndex()]->getSemaphore(),
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
}

bool Graph::render() {
  if (_resourcesByPass.empty()) return false;

  _tracer->markFrame();
  const auto cpuTrace = _tracer->traceCPU("Graph.render");
  const int frameSlot = getFrameInFlight();
  if (_frameNumber >= _maxFramesInFlight) {
    // Wait for every queue that used this slot, without joining independent GPU queues.
    const auto waitCpuTrace = _tracer->traceCPU("Graph.waitForFrame");
    std::vector<VkSemaphore> semaphores;
    std::vector<uint64_t> values;
    for (const auto& completion : _frameSlotCompletions[frameSlot]) {
      semaphores.push_back(completion.semaphore);
      values.push_back(completion.value);
    }
    const VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = static_cast<uint32_t>(semaphores.size()),
        .pSemaphores = semaphores.data(),
        .pValues = values.data(),
    };
    const auto result = vkWaitSemaphores(_device->getLogicalDevice(), &waitInfo, UINT64_MAX);
    if (result != VK_SUCCESS) throw std::runtime_error("vkWaitSemaphores failed: " + std::to_string(result));
  }

  if (_swapchain) {
    const auto result = _swapchain->acquireNextImage(*_swapchainImageAvailableSemaphores[frameSlot]);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) return true;
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("failed to acquire swap chain image");
  }

  // Record independent passes in parallel; plan each pass once its recording finishes.
  std::vector<std::future<void>> passRecordingTasks;
  passRecordingTasks.reserve(_passes.size());
  try {
    for (const auto& graphPass : _passes) {
      auto* currentPass = graphPass.get();
      auto* passCommandBuffer = &currentPass->getCommandBuffer(frameSlot);
      if (!passCommandBuffer->getActive()) passCommandBuffer->beginCommands();
      const QueueType passQueueType = currentPass->getQueueType();
      if (_firstPassesPerQueue.contains(currentPass)) _tracer->collect(passQueueType, *passCommandBuffer);
      passRecordingTasks.push_back(
          _threadPool->submit([this, currentPass, passCommandBuffer, passQueueType, frameSlot]() {
            const auto passCpuTrace = _tracer->traceCPU("Graph.pass." + currentPass->getName());
            const auto passGpuTrace = _tracer->traceGPU("Graph.pass." + currentPass->getName(), passQueueType,
                                                        *passCommandBuffer);
            currentPass->execute(frameSlot, *passCommandBuffer);
          }));
    }
    _preparePhysicalAccesses(passRecordingTasks);
  } catch (...) {
    // Also cover failures while scheduling tasks; no recording task may outlive render().
    for (auto& recordingTask : passRecordingTasks) {
      if (recordingTask.valid()) recordingTask.wait();
    }
    throw;
  }

  if (_swapchain) {
    auto& commands = _passes.back()->getCommandBuffer(frameSlot);
    const auto transitionTrace = _tracer->traceGPU("Graph.transition.present", _passes.back()->getQueueType(),
                                                   commands);
    auto& image = _swapchain->getImage(_swapchain->getSwapchainIndex());
    Barrier presentBarrier;
    presentBarrier.addImage(image.getImage(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_NONE, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            {image.getAspectMask(), 0, static_cast<uint32_t>(image.getMipMapNumber()), 0,
                             static_cast<uint32_t>(image.getLayerNumber())});
    presentBarrier.execute(commands.getCommandBuffer());
    image.setImageLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    // The final transition becomes the last write, so the next acquire follows it.
    if (auto history = _imageHistory.find(image.getImage()); history != _imageHistory.end()) {
      const auto queue = _frameSubmissions.back().queue;
      const auto& timeline = _queueTimelines.at(queue);
      history->second.accesses.assign(1, {.queue = queue,
                                          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                          .accessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                          .completion = {timeline.semaphore->getSemaphore(), timeline.lastSignalValue},
                                          .image = &image});
    }
  }

  std::vector<VkCommandBufferSubmitInfo> submissionCommandBuffers;
  for (auto& currentSubmission : _frameSubmissions) {
    submissionCommandBuffers.clear();
    submissionCommandBuffers.reserve(2 * (currentSubmission.endPassIndex - currentSubmission.firstPassIndex));
    for (auto passIndex = currentSubmission.firstPassIndex; passIndex < currentSubmission.endPassIndex; ++passIndex) {
      auto& pass = *_passes[passIndex];
      const auto& barriers = _barriersBeforePass.at(&pass);
      if (!barriers.getBufferBarriers().empty() || !barriers.getImageBarriers().empty()) {
        auto& commands = _barrierCommandBuffers.at(&pass).at(frameSlot);
        commands->beginCommands();
        barriers.execute(commands->getCommandBuffer());
        commands->endCommands();
        // Execute graph barriers before commands recorded by both update() and draw().
        submissionCommandBuffers.push_back(
            {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = commands->getCommandBuffer()});
      }
      auto& passCommandBuffer = _passes[passIndex]->getCommandBuffer(frameSlot);
      passCommandBuffer.endCommands();
      submissionCommandBuffers.push_back({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                          .commandBuffer = passCommandBuffer.getCommandBuffer()});
    }
    // Waits/signals apply to this submission regardless of when their descriptions were appended above.
    const VkSubmitInfo2 submissionInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(currentSubmission.semaphoreWaits.size()),
        .pWaitSemaphoreInfos = currentSubmission.semaphoreWaits.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(submissionCommandBuffers.size()),
        .pCommandBufferInfos = submissionCommandBuffers.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(currentSubmission.semaphoreSignals.size()),
        .pSignalSemaphoreInfos = currentSubmission.semaphoreSignals.data(),
    };
    const auto submissionResult = vkQueueSubmit2(currentSubmission.queue, 1, &submissionInfo, VK_NULL_HANDLE);
    if (submissionResult != VK_SUCCESS)
      throw std::runtime_error("vkQueueSubmit2 failed: " + std::to_string(submissionResult));
  }
  // Save actual submitted completion points; later submissions may advance these queue timelines again.
  auto& frameSlotCompletions = _frameSlotCompletions[frameSlot];
  frameSlotCompletions.clear();
  for (const auto& [graphQueue, queueTimeline] : _queueTimelines) {
    frameSlotCompletions.push_back({queueTimeline.semaphore->getSemaphore(), queueTimeline.lastSignalValue});
  }
  ++_frameNumber;
  if (!_swapchain) return false;

  const auto imageIndex = _swapchain->getSwapchainIndex();
  const auto finished = _swapchainRenderFinishedSemaphores[imageIndex]->getSemaphore();
  const VkSwapchainKHR swapchain = _swapchain->getSwapchain();
  const VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &finished,
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &imageIndex,
  };
  const auto result = vkQueuePresentKHR(_device->getQueue(QueueType::GENERAL), &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) return true;
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    throw std::runtime_error("failed to present swap chain image: " + std::to_string(result));
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

  // Images may be destroyed by storage reset. Keep their readiness as global waits, without old Image pointers.
  {
    const std::scoped_lock lock(_pendingWaitsMutex);
    for (auto& wait : _pendingWaits) wait.image = nullptr;
  }
  for (const auto& [imageHandle, resourceHistory] : _imageHistory) {
    for (const auto& previousAccess : resourceHistory.accesses) {
      if (previousAccess.queue == VK_NULL_HANDLE) addFrameWait(previousAccess.completion);
    }
  }
  _imageHistory.clear();

  auto oldSwapchainImageViews = _swapchain->reset(resolution);
  _graphStorage->reset(oldSwapchainImageViews, _swapchain->getImageViews());

  _swapchainRenderFinishedSemaphores.clear();
  std::ranges::generate_n(std::back_inserter(_swapchainRenderFinishedSemaphores), _swapchain->getImageCount(),
                          [&] { return std::make_shared<Semaphore>(VK_SEMAPHORE_TYPE_BINARY, *_device); });

  for (const auto& graphPass : _passes) {
    graphPass->reset(_swapchain->getImageViews());
  }

  calculate();
}
