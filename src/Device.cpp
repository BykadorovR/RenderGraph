module;

#include <VkBootstrap.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <volk.h>

module Device;

using namespace RenderGraph;

Device::Device(const Surface& surface, const Instance& instance) {
  _instance = &instance;
  _surface = &surface;
}

void Device::initialize() {
  // Vulkan 1.0 features
  VkPhysicalDeviceFeatures deviceFeatures{
      .geometryShader = true,
      .tessellationShader = true,
      .samplerAnisotropy = true,
  };

  // Vulkan 1.2 features
  VkPhysicalDeviceVulkan12Features features12{};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features12.drawIndirectCount = true;
  features12.hostQueryReset = true;
  features12.timelineSemaphore = true;
  features12.bufferDeviceAddress = true;

  // Vulkan 1.3 features
  VkPhysicalDeviceVulkan13Features features13{};
  features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features13.synchronization2 = true;
  features13.dynamicRendering = true;

  // Not part of Vulkan 1.3 core
  VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .descriptorBuffer = true};

  vkb::PhysicalDeviceSelector deviceSelector(_instance->getInstance());
  // vk-bootstrap propagates features used as physical-device selection
  // criteria into logical-device creation, so builder.add_pNext is not
  // needed for these core features.
  deviceSelector.set_required_features(deviceFeatures);
  deviceSelector.set_required_features_12(features12);
  deviceSelector.set_required_features_13(features13);
  deviceSelector.allow_any_gpu_device_type(false);
  // not part of Vulkan 1.3 core
  if (std::find(_optionalExtensions.begin(), _optionalExtensions.end(), "VK_EXT_descriptor_buffer") !=
      _optionalExtensions.end())
    deviceSelector.add_desired_extension("VK_EXT_descriptor_buffer");
  // VK_KHR_SWAPCHAIN_EXTENSION_NAME is added by default
  deviceSelector.set_surface(_surface->getSurface());
  auto deviceSelectorResult = deviceSelector.select();
  if (!deviceSelectorResult) {
    throw std::runtime_error(deviceSelectorResult.error().message());
  }
  auto devicePhysical = deviceSelectorResult.value();

  vkb::DeviceBuilder builder{devicePhysical};
  if (devicePhysical.is_extension_present("VK_EXT_descriptor_buffer") &&
      std::find(_optionalExtensions.begin(), _optionalExtensions.end(), "VK_EXT_descriptor_buffer") !=
          _optionalExtensions.end())
    builder.add_pNext(&descriptorBufferFeatures);

  auto builderResult = builder.build();
  if (!builderResult) {
    throw std::runtime_error(builderResult.error().message());
  }
  _device = builderResult.value();
  volkLoadDevice(_device.device);

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, nullptr);
  _queueFamilyProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, _queueFamilyProperties.data());
}

void Device::setOptionalExtensions(const std::vector<std::string>& extensions) {
  _optionalExtensions = extensions;
}

const VkQueueFamilyProperties& Device::getQueueFamilyProperties(vkb::QueueType type) const {
  return _queueFamilyProperties[getQueueIndex(type)];
}

bool Device::isExtensionSupported(std::string name) const {
  return _device.physical_device.is_extension_present(name.c_str());
}

std::vector<std::string> Device::getOptionalExtensions() const { return _optionalExtensions; }

bool Device::isFormatFeatureSupported(VkFormat format,
                                      VkImageTiling tiling,
                                      VkFormatFeatureFlagBits featureFlagBit) const {
  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(getPhysicalDevice(), format, &props);

  if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & featureFlagBit) == featureFlagBit) {
    return true;
  } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & featureFlagBit) == featureFlagBit) {
    return true;
  }

  return false;
}

const VkDevice Device::getLogicalDevice() const noexcept { return _device.device; }

const VkPhysicalDevice Device::getPhysicalDevice() const noexcept { return _device.physical_device.physical_device; }

const vkb::Device& Device::getDevice() const noexcept { return _device; }

const VkQueue Device::getQueue(vkb::QueueType type) const {
  auto queueResult = _device.get_dedicated_queue(type);
  if (!queueResult) {
    queueResult = _device.get_queue(type);
    // use default queue that should support everything
    if (!queueResult) {
      queueResult = _device.get_queue(vkb::QueueType::present);
    }
  }

  return queueResult.value();
}

int Device::getQueueIndex(vkb::QueueType type) const {
  auto queueResult = _device.get_dedicated_queue_index(type);
  if (!queueResult) {
    queueResult = _device.get_queue_index(type);
    // use default queue that should support everything
    if (!queueResult) {
      queueResult = _device.get_queue_index(vkb::QueueType::present);
    }
  }

  return queueResult.value();
}

Device::~Device() { vkb::destroy_device(_device); }
