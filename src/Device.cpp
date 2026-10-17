module;

#include <VkBootstrap.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <volk.h>

module Device;

using namespace RenderGraph;

Device::Device(const Instance& instance) : _instance(&instance) {}

void Device::setSurface(const Surface& surface) {
  if (_initialized) {
    throw std::logic_error("Can't set a surface after device initialization");
  }
  _surface = &surface;
}

void Device::setPhysicalDevice(VkPhysicalDevice physicalDevice) {
  if (_initialized) {
    throw std::logic_error("Can't set a physical device after device initialization");
  }
  _selectedPhysicalDevice = physicalDevice;
}

std::vector<std::pair<VkPhysicalDevice, std::string>> Device::getAvailablePhysicalDevices() const {
  uint32_t physicalDeviceCount = 0;
  if (vkEnumeratePhysicalDevices(_instance->getInstance().instance, &physicalDeviceCount, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to enumerate physical devices");
  }

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  if (vkEnumeratePhysicalDevices(_instance->getInstance().instance, &physicalDeviceCount, physicalDevices.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to enumerate physical devices");
  }

  std::vector<std::pair<VkPhysicalDevice, std::string>> devices;
  devices.reserve(physicalDevices.size());
  for (const auto physicalDevice : physicalDevices) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    devices.emplace_back(physicalDevice, properties.deviceName);
  }
  return devices;
}

void Device::initialize(const DeviceRequirements& requirements) {
  if (_initialized) {
    throw std::logic_error("Device is already initialized");
  }

  const auto descriptorBufferRequired = std::find(requirements.requiredExtensions.begin(),
                                                  requirements.requiredExtensions.end(),
                                                  VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) !=
                                        requirements.requiredExtensions.end();
  const auto descriptorBufferOptional = std::find(requirements.optionalExtensions.begin(),
                                                  requirements.optionalExtensions.end(),
                                                  VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) !=
                                        requirements.optionalExtensions.end();

  // Vulkan 1.2 features
  VkPhysicalDeviceVulkan12Features features12{};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features12.hostQueryReset = true;
  features12.timelineSemaphore = true;
  features12.bufferDeviceAddress = true;

  // Vulkan 1.3 features
  VkPhysicalDeviceVulkan13Features features13{};
  features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features13.synchronization2 = true;
  features13.dynamicRendering = true;

  vkb::PhysicalDeviceSelector deviceSelector(_instance->getInstance());
  // vk-bootstrap propagates features used as physical-device selection
  // criteria into logical-device creation, so builder.add_pNext is not
  // needed for these core features.
  deviceSelector.set_required_features_12(features12);
  deviceSelector.set_required_features_13(features13);
  deviceSelector.set_required_features(requirements.features);
  deviceSelector.set_required_features_12(requirements.features12);
  deviceSelector.set_required_features_13(requirements.features13);

  for (const auto& extension : requirements.requiredExtensions) {
    deviceSelector.add_required_extension(extension.c_str());
  }
  for (const auto& extension : requirements.optionalExtensions) {
    if (extension != VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) {
      deviceSelector.add_desired_extension(extension.c_str());
    }
  }
  if (descriptorBufferRequired && requirements.descriptorBufferFeatures.descriptorBuffer) {
    auto descriptorBufferFeatures = requirements.descriptorBufferFeatures;
    descriptorBufferFeatures.pNext = nullptr;
    deviceSelector.add_required_extension_features(descriptorBufferFeatures);
  }
  deviceSelector.allow_any_gpu_device_type(false);
  // vk-bootstrap enables VK_KHR_swapchain when a surface is provided.
  if (_surface != nullptr) {
    deviceSelector.set_surface(_surface->getSurface());
  } else {
    deviceSelector.require_present(false);
  }

  vkb::PhysicalDevice devicePhysical;
  if (_selectedPhysicalDevice == VK_NULL_HANDLE) {
    auto deviceResult = deviceSelector.select();
    if (!deviceResult) {
      throw std::runtime_error(deviceResult.error().message());
    }
    devicePhysical = deviceResult.value();
  } else {
    auto devicesResult = deviceSelector.select_devices();
    if (!devicesResult) {
      throw std::runtime_error(devicesResult.error().message());
    }
    const auto selectedDevice = std::find_if(
        devicesResult->begin(), devicesResult->end(),
        [this](const vkb::PhysicalDevice& device) { return device.physical_device == _selectedPhysicalDevice; });
    if (selectedDevice == devicesResult->end()) {
      throw std::runtime_error("Selected physical device does not satisfy the device requirements");
    }
    devicePhysical = *selectedDevice;
  }
  const auto enableOptionalDescriptorBuffer =
      !descriptorBufferRequired && descriptorBufferOptional && requirements.descriptorBufferFeatures.descriptorBuffer &&
      devicePhysical.is_extension_present(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
  if (enableOptionalDescriptorBuffer) {
    auto descriptorBufferFeatures = requirements.descriptorBufferFeatures;
    descriptorBufferFeatures.pNext = nullptr;
    if (devicePhysical.enable_extension_features_if_present(descriptorBufferFeatures)) {
      devicePhysical.enable_extension_if_present(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    }
  }
  _enabledExtensions = devicePhysical.get_extensions();

  vkb::DeviceBuilder builder{devicePhysical};
  for (auto* extensionFeatures = requirements.extensionFeatures; extensionFeatures != nullptr;
       extensionFeatures = extensionFeatures->pNext) {
    builder.add_pNext(extensionFeatures);
  }
  auto builderResult = builder.build();
  if (!builderResult) {
    throw std::runtime_error(builderResult.error().message());
  }
  _device = builderResult.value();
  _initialized = true;
  volkLoadDevice(_device.device);

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, nullptr);
  _queueFamilyProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, _queueFamilyProperties.data());
}

const VkQueueFamilyProperties& Device::getQueueFamilyProperties(QueueType type) const {
  return _queueFamilyProperties[getQueueIndex(type)];
}

bool Device::isExtensionSupported(std::string name) const {
  return _device.physical_device.is_extension_present(name.c_str());
}

bool Device::isExtensionEnabled(std::string name) const {
  return std::find(_enabledExtensions.begin(), _enabledExtensions.end(), name) != _enabledExtensions.end();
}

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

const VkQueue Device::getQueue(QueueType type) const {
  vkb::QueueType bootstrapType;
  switch (type) {
    case QueueType::PRESENT:
      bootstrapType = vkb::QueueType::present;
      break;
    case QueueType::GRAPHICS:
      bootstrapType = vkb::QueueType::graphics;
      break;
    case QueueType::COMPUTE:
      bootstrapType = vkb::QueueType::compute;
      break;
    case QueueType::TRANSFER:
      bootstrapType = vkb::QueueType::transfer;
      break;
  }
  auto queueResult = _device.get_dedicated_queue(bootstrapType);
  if (!queueResult) {
    queueResult = _device.get_queue(bootstrapType);
    // use default queue that should support everything
    if (!queueResult) {
      queueResult = _device.get_queue(vkb::QueueType::graphics);
    }
  }

  return queueResult.value();
}

int Device::getQueueIndex(QueueType type) const {
  vkb::QueueType bootstrapType;
  switch (type) {
    case QueueType::PRESENT:
      bootstrapType = vkb::QueueType::present;
      break;
    case QueueType::GRAPHICS:
      bootstrapType = vkb::QueueType::graphics;
      break;
    case QueueType::COMPUTE:
      bootstrapType = vkb::QueueType::compute;
      break;
    case QueueType::TRANSFER:
      bootstrapType = vkb::QueueType::transfer;
      break;
  }
  auto queueResult = _device.get_dedicated_queue_index(bootstrapType);
  if (!queueResult) {
    queueResult = _device.get_queue_index(bootstrapType);
    // use default queue that should support everything
    if (!queueResult) {
      queueResult = _device.get_queue_index(vkb::QueueType::graphics);
    }
  }

  return queueResult.value();
}

Device::~Device() {
  if (_initialized) {
    vkb::destroy_device(_device);
  }
}
