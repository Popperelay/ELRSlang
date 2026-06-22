#include <jni.h>
#include <string>
#include <vulkan/vulkan.h>

extern "C" JNIEXPORT jstring JNICALL
Java_dev_elrslang_SmokeActivity_nativeSmokeStatus(JNIEnv* env, jclass) {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ELRSlang Smoke";
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
    if (result == VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return env->NewStringUTF("ELRSlang Android Vulkan smoke: OK");
    }

    std::string message = "ELRSlang Android Vulkan smoke failed: VkResult ";
    message += std::to_string(static_cast<int>(result));
    return env->NewStringUTF(message.c_str());
}
