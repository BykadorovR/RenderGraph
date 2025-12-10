module Instance;
using namespace RenderGraph;

VkBool32 debugCallbackUtils(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageType,
                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                            void* pUserData) {
  if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
    printf("\033[32m[Validation layer] %s\033[0m\n", pCallbackData->pMessage);
  else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
    printf("\033[33m[Validation layer] %s\033[0m\n", pCallbackData->pMessage);
  else
    printf("\033[31m[Validation layer] %s\033[0m\n", pCallbackData->pMessage);
  return false;
}

Instance::Instance(std::string_view name, bool validation) {
  auto sts = volkInitialize();
  if (sts != VK_SUCCESS) throw std::runtime_error("Can't initialize Vulkan Loader");
  // VK_KHR_win32_surface || VK_KHR_android_surface as well as VK_KHR_surface
  // are automatically added by vkb
  vkb::InstanceBuilder builder;
  auto systemInfoResult = vkb::SystemInfo::get_system_info();
  if (!systemInfoResult) throw std::runtime_error(systemInfoResult.error().message());
  auto systemInfo = systemInfoResult.value();
  // debug utils is a newer API, can be not supported. Includes validation layers + API for code instrumentation.
  // debug report is deprecated, includes only validation layers.
  if (validation && systemInfo.validation_layers_available) {
    builder.enable_validation_layers();
    builder.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
    builder.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);    
    if (systemInfo.debug_utils_available) {
      builder.set_debug_messenger_type(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
      builder.set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
      builder.set_debug_callback(debugCallbackUtils);
      _debugUtils = true;
    }
  }
  auto instanceResult = builder.set_app_name(name.data()).require_api_version(1, 3, 0).build();
  if (!instanceResult)
    throw std::runtime_error("Failed to create Vulkan instance. Error: " + instanceResult.error().message());

  _instance = instanceResult.value();
  volkLoadInstance(_instance.instance);
}

bool Instance::isDebug() const noexcept { return _debugUtils; }

const vkb::Instance& Instance::getInstance() const noexcept { return _instance; }

Instance::~Instance() { vkb::destroy_instance(_instance); }