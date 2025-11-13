"""
GPU Availability Checker
Проверяет доступность GPU для TensorFlow
"""

import sys

print("=" * 80)
print("🔍 ПРОВЕРКА GPU")
print("=" * 80)

# 1. Проверяем TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow установлен: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow не установлен!")
    print("   Установите: pip install tensorflow>=2.12.0")
    sys.exit(1)

# 2. Проверяем GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n📊 Найдено GPU устройств: {len(gpus)}")

if len(gpus) > 0:
    print("\n✅ GPU ДОСТУПЕН!")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        # Получаем детали
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"      Детали: {gpu_details}")
        except:
            pass
    
    # Проверяем CUDA
    print(f"\n🔧 CUDA доступна: {tf.test.is_built_with_cuda()}")
    print(f"🔧 GPU доступен для использования: {tf.test.is_gpu_available()}")
    
    # Тест производительности
    print("\n⚡ Тест производительности...")
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([10000, 10000])
            b = tf.random.normal([10000, 10000])
            import time
            start = time.time()
            c = tf.matmul(a, b)
            gpu_time = time.time() - start
            print(f"   GPU время: {gpu_time:.4f} сек")
    except Exception as e:
        print(f"   ⚠️ Ошибка GPU теста: {e}")
    
    # CPU тест для сравнения
    try:
        with tf.device('/CPU:0'):
            a = tf.random.normal([10000, 10000])
            b = tf.random.normal([10000, 10000])
            start = time.time()
            c = tf.matmul(a, b)
            cpu_time = time.time() - start
            print(f"   CPU время: {cpu_time:.4f} сек")
            print(f"   🚀 Ускорение: {cpu_time/gpu_time:.2f}x")
    except Exception as e:
        print(f"   ⚠️ Ошибка CPU теста: {e}")
        
else:
    print("\n❌ GPU НЕ НАЙДЕН")
    print("\n📝 Для использования GPU:")
    print("   1. Убедитесь, что у вас есть NVIDIA GPU")
    print("   2. Установите CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("   3. Установите cuDNN: https://developer.nvidia.com/cudnn")
    print("   4. Установите TensorFlow с GPU:")
    print("      pip uninstall tensorflow")
    print("      pip install tensorflow[and-cuda]  # Для TensorFlow 2.16+")
    print("      или")
    print("      pip install tensorflow-gpu  # Для более старых версий")

print("\n" + "=" * 80)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 80)




