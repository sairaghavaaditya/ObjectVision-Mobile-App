import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { API_BASE, getMimeFromUri } from '@/constants/Api';
import { useThemeColor } from '@/hooks/useThemeColor';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  SafeAreaView,
  StyleSheet,
  TouchableOpacity,
  View
} from 'react-native';
import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImageSize, setSelectedImageSize] = useState<{ width: number; height: number } | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<string | null>(null);
  const router = useRouter();
  
  const backgroundColor = useThemeColor({}, 'background');
  const textColor = useThemeColor({}, 'text');
  const tintColor = useThemeColor({}, 'tint');

  function encodeBase64(bytes: Uint8Array): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';
    let output = '';
    let i = 0;
    while (i < bytes.length) {
      const byte1 = bytes[i++] ?? 0;
      const byte2 = bytes[i++] ?? 0;
      const byte3 = bytes[i++] ?? 0;

      const enc1 = byte1 >> 2;
      const enc2 = ((byte1 & 3) << 4) | (byte2 >> 4);
      const enc3 = ((byte2 & 15) << 2) | (byte3 >> 6);
      const enc4 = byte3 & 63;

      const isByte2Missing = i - 1 > bytes.length;
      const isByte3Missing = i > bytes.length;

      output += chars.charAt(enc1);
      output += chars.charAt(enc2);
      output += bytes.length + 1 < i ? '=' : chars.charAt(enc3);
      output += bytes.length < i ? '=' : chars.charAt(enc4);
    }
    return output;
  }

  const requestPermissions = async () => {
    if (Platform.OS === 'web') return true;
    const camera = await ImagePicker.requestCameraPermissionsAsync();
    const media = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (camera.status !== 'granted' || media.status !== 'granted') {
      Alert.alert('Permissions Required', 'Camera and media library permissions are required to use this app.', [{ text: 'OK' }]);
      return false;
    }
    return true;
  };

  const handleImageSelection = async (source: 'camera' | 'gallery') => {
    const hasPermissions = await requestPermissions();
    if (!hasPermissions) return;

    setIsLoading(true);

    try {
      if (Platform.OS === 'web') {
        // Web fallback: use a hidden file input for reliable selection
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        if (source === 'camera') (input as any).capture = 'environment';
        input.onchange = async () => {
          const file = input.files && input.files[0];
          if (!file) {
            setIsLoading(false);
            return;
          }
          const objectUrl = URL.createObjectURL(file);
          setSelectedImage(objectUrl);
          try {
            // Attempt to read dimensions
            const img = new Image();
            img.onload = () => {
              setSelectedImageSize({ width: img.width, height: img.height });
              URL.revokeObjectURL(objectUrl);
            };
            img.src = objectUrl;
          } catch {}
          setIsLoading(false);
        };
        input.click();
        return;
      }

      let result: ImagePicker.ImagePickerResult;
      if (source === 'camera') {
        result = await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images as any,
          allowsEditing: false,
          quality: 1,
        });
      } else {
        result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images as any,
          allowsEditing: false,
          quality: 1,
        });
      }

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        setSelectedImage(asset.uri);
        if (asset.width && asset.height) setSelectedImageSize({ width: asset.width, height: asset.height });
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to select image. Please try again.', [{ text: 'OK' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDetectObjects = async () => {
    if (!selectedImage) {
      Alert.alert('No image', 'Please select an image from Camera or Gallery first.', [{ text: 'OK' }]);
      return;
    }

    setIsDetecting(true);
    setLastError(null);
    
    const fetchWithTimeout = async (input: RequestInfo | URL, init: RequestInit & { timeoutMs?: number } = {}) => {
      const { timeoutMs = 25000, ...rest } = init;
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const res = await fetch(input, { ...rest, signal: controller.signal });
        return res;
      } finally {
        clearTimeout(id);
      }
    };

    try {
      const confidence = 0.3;
      const detectSize = 640; // resize long side for better boxes/speed
      const classifyMode = 'warp'; // full-image scaling for better generic labels
      const annotateUrl = `${API_BASE}/detect-image?confidence=${confidence}&detect_size=${detectSize}&classify_mode=${classifyMode}`;

      let response: Response;

      if (Platform.OS === 'web') {
        const picked = await fetch(selectedImage);
        const fileBlob = await picked.blob();
        const formData = new FormData();
        const mime = fileBlob.type || 'image/jpeg';
        formData.append('file', new File([fileBlob], 'image.jpg', { type: mime }));
        response = await fetchWithTimeout(annotateUrl, {
          method: 'POST',
          headers: { Accept: 'image/png' },
          body: formData,
          timeoutMs: 30000,
        });
      } else {
        const mime = getMimeFromUri(selectedImage);
        const ext = mime.split('/')[1] || 'jpg';
        const filename = `image.${ext}`;
        const form = new FormData();
        form.append('file', { uri: selectedImage, name: filename, type: mime } as any);
        response = await fetchWithTimeout(annotateUrl, {
          method: 'POST',
          headers: { Accept: 'image/png' },
          body: form,
          timeoutMs: 30000,
        });
      }

      if (!response.ok) {
        const contentType = response.headers.get('content-type') || '';
        let message = 'Detection failed';
        try {
          if (contentType.includes('application/json')) {
            const json = await response.json();
            message = json.detail || json.message || JSON.stringify(json);
          } else {
            const text = await response.text();
            if (text) message = text;
          }
        } catch {}
        throw new Error(message);
      }

      const detectionsCount = response.headers.get('X-Detections-Count') || '0';
      const processingTime = response.headers.get('X-Processing-Time') || '';
      const originalWidth = response.headers.get('X-Image-Width') || '';
      const originalHeight = response.headers.get('X-Image-Height') || '';
      let clsLabel = response.headers.get('X-Cls-Label') || '';
      let clsProb = response.headers.get('X-Cls-Prob') || '';

      // If classification headers not present, call /classify as a fallback
      if (!clsLabel) {
        try {
          const classifyUrl = `${API_BASE}/classify`;
          if (Platform.OS === 'web') {
            const picked = await fetch(selectedImage);
            const fileBlob = await picked.blob();
            const formData = new FormData();
            const mime = fileBlob.type || 'image/jpeg';
            formData.append('file', new File([fileBlob], 'image.jpg', { type: mime }));
            const classifyRes = await fetch(classifyUrl, { method: 'POST', body: formData });
            if (classifyRes.ok) {
              const json = await classifyRes.json();
              clsLabel = json?.classification?.top1?.label || '';
              const p = json?.classification?.top1?.probability;
              clsProb = typeof p === 'number' ? String(p) : '';
            }
          } else {
            const mime = getMimeFromUri(selectedImage);
            const ext = mime.split('/')[1] || 'jpg';
            const filename = `image.${ext}`;
            const form2 = new FormData();
            form2.append('file', { uri: selectedImage, name: filename, type: mime } as any);
            const classifyRes = await fetch(classifyUrl, { method: 'POST', body: form2 });
            if (classifyRes.ok) {
              const json = await classifyRes.json();
              clsLabel = json?.classification?.top1?.label || '';
              const p = json?.classification?.top1?.probability;
              clsProb = typeof p === 'number' ? String(p) : '';
            }
          }
        } catch {}
      }

      const blob = await response.blob();
      const dataUrl: string = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(typeof reader.result === 'string' ? reader.result : '');
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      let annotatedUriToPass = dataUrl;
      if (Platform.OS !== 'web') {
        try {
          const base64Data = dataUrl.split(',')[1] || '';
          const fileUri = `${FileSystem.cacheDirectory}annotated_${Date.now()}.png`;
          await FileSystem.writeAsStringAsync(fileUri, base64Data, { encoding: FileSystem.EncodingType.Base64 });
          annotatedUriToPass = fileUri;
        } catch {}
      }

      router.push({
        pathname: '/results',
        params: {
          imageUri: selectedImage,
          annotatedUri: annotatedUriToPass,
          detectionsCount,
          processingTime,
          originalWidth,
          originalHeight,
          clsLabel,
          clsProb,
          // Provide a caption fallback as a convenience for the results screen
          caption: clsLabel || '',
        },
      });
    } catch (error) {
      let message = 'Failed to detect objects. Please try again.';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          message = 'Request timed out. Ensure backend is running on the same host/port.';
        } else if (/Network request failed|ERR_CONNECTION/i.test(error.message)) {
          message = 'Network error. Ensure backend is reachable at ' + API_BASE + ' and CORS is enabled.';
        } else {
          message = error.message;
        }
      }
      setLastError(message);
      Alert.alert('Error', message, [{ text: 'OK' }]);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleTestConnection = async () => {
    setHealthStatus(null);
    setLastError(null);
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
      if (!res.ok) throw new Error(`Health check failed (${res.status})`);
      const json = await res.json();
      setHealthStatus(`OK • model_loaded=${json.model_loaded} • device=${json.device}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to reach backend';
      setLastError(`Health error: ${msg}`);
    }
  };

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={tintColor} />
          <ThemedText style={styles.loadingText}>Loading image...</ThemedText>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor }]}>
      <ThemedView style={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <ThemedText type="title" style={styles.title}>
            Object Detection
          </ThemedText>
          <ThemedText style={styles.subtitle}>
            Take a photo or select from gallery
          </ThemedText>
        </View>

        {/* Camera and Gallery Buttons */}
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[styles.sourceButton, { borderColor: tintColor }]}
            onPress={() => handleImageSelection('camera')}
            activeOpacity={0.8}
          >
            <Ionicons name="camera" size={24} color={tintColor} />
            <ThemedText style={[styles.sourceButtonText, { color: tintColor }]}>Camera</ThemedText>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.sourceButton, { borderColor: tintColor }]}
            onPress={() => handleImageSelection('gallery')}
            activeOpacity={0.8}
          >
            <Ionicons name="images" size={24} color={tintColor} />
            <ThemedText style={[styles.sourceButtonText, { color: tintColor }]}>Gallery</ThemedText>
          </TouchableOpacity>
        </View>

        {/* Diagnostics */}
        <View style={{ marginBottom: 10 }}>
          <ThemedText style={{ fontSize: 12, opacity: 0.8 }}>API: {API_BASE}</ThemedText>
          {healthStatus ? (
            <ThemedText style={{ fontSize: 12, color: '#2e7d32' }}>{healthStatus}</ThemedText>
          ) : null}
          {lastError ? (
            <ThemedText style={{ fontSize: 12, color: '#b00020' }}>{lastError}</ThemedText>
          ) : null}
          <TouchableOpacity
            style={{ marginTop: 6, alignSelf: 'flex-start', paddingVertical: 6, paddingHorizontal: 10, borderRadius: 8, borderWidth: 1, borderColor: '#ccc' }}
            onPress={handleTestConnection}
            activeOpacity={0.8}
          >
            <ThemedText>Test Connection</ThemedText>
          </TouchableOpacity>
        </View>

        {/* Image Preview */}
        <View style={styles.imagePreviewContainer}>
          {selectedImage ? (
            <Image
              source={{ uri: selectedImage }}
              style={styles.previewImage}
              contentFit="contain"
            />
          ) : (
            <View style={styles.placeholderContainer}>
              <Ionicons name="image-outline" size={48} color="#ccc" />
              <ThemedText style={styles.placeholderText}>No image selected</ThemedText>
            </View>
          )}
        </View>
        {selectedImage ? (
          <ThemedText style={{ fontSize: 12, opacity: 0.6, marginBottom: 10 }}>Selected: {selectedImage}</ThemedText>
        ) : null}

        {/* Detect Objects Button */}
        <TouchableOpacity
          style={[styles.detectButton, { backgroundColor: '#20B2AA' }]}
          onPress={handleDetectObjects}
          disabled={!selectedImage || isDetecting}
          activeOpacity={0.8}
        >
          {isDetecting ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Ionicons name="cloud-upload" size={20} color="white" />
          )}
          <ThemedText style={styles.detectButtonText}>
            {isDetecting ? 'Detecting...' : 'Detect Objects'}
          </ThemedText>
        </TouchableOpacity>
      </ThemedView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.7,
    textAlign: 'center',
    lineHeight: 22,
    color: '#666',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 15,
  },
  sourceButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    backgroundColor: 'transparent',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 60,
  },
  sourceButtonText: {
    fontSize: 16,
    fontWeight: '600',
    marginTop: 8,
  },
  imagePreviewContainer: {
    flex: 1,
    borderRadius: 12,
    backgroundColor: '#f8f8f8',
    marginBottom: 20,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  placeholderText: {
    marginTop: 12,
    fontSize: 16,
    color: '#999',
    textAlign: 'center',
  },
  detectButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    gap: 8,
  },
  detectButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
  },
  loadingText: {
    fontSize: 16,
    opacity: 0.7,
  },
});
