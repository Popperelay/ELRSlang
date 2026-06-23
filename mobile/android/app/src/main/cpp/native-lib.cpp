#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <string>
#include <vulkan/vulkan.h>

static bool assetExists(AAssetManager* assets, const char* path) {
    if (!assets) return false;
    AAsset* asset = AAssetManager_open(assets, path, AASSET_MODE_BUFFER);
    if (!asset) return false;
    AAsset_close(asset);
    return true;
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_elrslang_SmokeActivity_nativeSmokeStatus(JNIEnv* env, jclass, jobject assetManager) {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ELRSlang Smoke";
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance);

    AAssetManager* assets = AAssetManager_fromJava(env, assetManager);
    bool hasManifest = assetExists(assets, "elrslang/manifest.json");
    bool hasGraph = assetExists(assets, "elrslang/graph.json");

    std::string message;
    if (result == VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        message = "ELRSlang Android Vulkan smoke: OK";
    } else {
        message = "ELRSlang Android Vulkan smoke failed: VkResult ";
        message += std::to_string(static_cast<int>(result));
    }
    message += hasManifest && hasGraph ? "\nAsset pack: OK" : "\nAsset pack: missing";
    return env->NewStringUTF(message.c_str());
}
