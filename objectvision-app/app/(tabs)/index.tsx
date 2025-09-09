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

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImageSize, setSelectedImageSize] = useState<{ width: number; height: number } | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
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
    const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
    const { status: mediaStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (cameraStatus !== 'granted' || mediaStatus !== 'granted') {
      Alert.alert(
        'Permissions Required',
        'Camera and media library permissions are required to use this app.',
        [{ text: 'OK' }]
      );
      return false;
    }
    return true;
  };

  const handleImageSelection = async (source: 'camera' | 'gallery') => {
    const hasPermissions = await requestPermissions();
    if (!hasPermissions) return;

    setIsLoading(true);

    try {
      let result;
      
      if (source === 'camera') {
        result = await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: false,
          quality: 1,
        });
      } else {
        result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
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
      Alert.alert(
        'Error',
        'Failed to select image. Please try again.',
        [{ text: 'OK' }]
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDetectObjects = async () => {
    if (!selectedImage) return;

    setIsDetecting(true);
    
    try {
      // Prepare multipart form data
      const form = new FormData();
      const mime = getMimeFromUri(selectedImage);
      const ext = mime.split('/')[1] || 'jpg';
      const filename = `image.${ext}`;
      form.append('file', {
        uri: selectedImage,
        name: filename,
        type: mime,
      } as any);

      const confidence = 0.3;
      const apiUrl = `${API_BASE}/detect-image?confidence=${confidence}`;
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          Accept: 'image/png',
        },
        body: form,
      });

      if (!response.ok) {
        const contentType = response.headers.get('content-type') || '';
        const text = await response.text();
        let message = 'Detection failed';
        try {
          if (contentType.includes('application/json')) {
            const json = JSON.parse(text);
            message = json.detail || JSON.stringify(json);
          } else if (text) {
            message = text;
          }
        } catch {}
        throw new Error(message);
      }

      const detectionsCount = response.headers.get('X-Detections-Count') || '0';
      const processingTime = response.headers.get('X-Processing-Time') || '';
      const originalWidth = response.headers.get('X-Image-Width') || '';
      const originalHeight = response.headers.get('X-Image-Height') || '';

      const buffer = await response.arrayBuffer();
      const base64 = encodeBase64(new Uint8Array(buffer));
      const annotatedImageUri = `data:image/png;base64,${base64}`;

      router.push({
        pathname: '/results',
        params: {
          imageUri: selectedImage,
          annotatedUri: annotatedImageUri,
          detectionsCount,
          processingTime,
          originalWidth,
          originalHeight,
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to detect objects. Please try again.';
      Alert.alert(
        'Error',
        message,
        [{ text: 'OK' }]
      );
    } finally {
      setIsDetecting(false);
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
            <ThemedText style={[styles.sourceButtonText, { color: tintColor }]}>
              Camera
            </ThemedText>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.sourceButton, { borderColor: tintColor }]}
            onPress={() => handleImageSelection('gallery')}
            activeOpacity={0.8}
          >
            <Ionicons name="images" size={24} color={tintColor} />
            <ThemedText style={[styles.sourceButtonText, { color: tintColor }]}>
              Gallery
            </ThemedText>
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
              <ThemedText style={styles.placeholderText}>
                No image selected
              </ThemedText>
            </View>
          )}
        </View>

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
