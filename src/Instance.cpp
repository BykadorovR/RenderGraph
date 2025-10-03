module Instance;
using namespace RenderGraph;

VkBool32 debugCallbackUtils(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageType,
                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                            void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
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
    if (systemInfo.debug_utils_available) {
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