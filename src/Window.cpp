module Window;
using namespace RenderGraph;
import "GLFW/glfw3.h";

bool Window::getResized() const noexcept { return _resized; }

void Window::setResized(bool resized) noexcept { _resized = resized; }

GLFWwindow* Window::getWindow() const noexcept { return _window; }

Window::Window(glm::ivec2 resolution) noexcept { _resolution = resolution; }

void Window::setFullScreen(bool fullScreen) {
  if (fullScreen) {
    auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    glfwSetWindowMonitor(_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
  } else {
    glfwSetWindowMonitor(_window, nullptr, 0, 0, _resolution.x, _resolution.y, GLFW_DONT_CARE);
  }
}

void Window::initialize() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  _window = glfwCreateWindow(_resolution.x, _resolution.y, "Vulkan", nullptr, nullptr);
  glfwSetWindowUserPointer(_window, this);
  glfwSetFramebufferSizeCallback(_window, [](GLFWwindow* window, int width, int height) {
    reinterpret_cast<Window*>(glfwGetWindowUserPointer(window))->setResized(true);
  });
}

Window::~Window() {
  glfwDestroyWindow(_window);
  glfwTerminate();
}