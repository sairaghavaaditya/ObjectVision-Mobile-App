import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { Image } from 'expo-image';
import { useLocalSearchParams } from 'expo-router';
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function ResultsScreen() {
  const { imageUri, annotatedUri, detectionsCount, processingTime, originalWidth, originalHeight } = useLocalSearchParams();
  const backgroundColor = useThemeColor({}, 'background');

  return (
    <SafeAreaView style={[styles.container, { backgroundColor }]}>
      <ThemedView style={styles.content}>
        <ThemedText type="title" style={styles.title}>
          Detection Results
        </ThemedText>
        
        <View style={styles.imageContainer}>
          {annotatedUri ? (
            <Image
              source={{ uri: String(annotatedUri) }}
              style={{ width: '100%', height: '100%' }}
              contentFit="contain"
            />
          ) : (
            <Image
              source={{ uri: String(imageUri) }}
              style={{ width: '100%', height: '100%' }}
              contentFit="contain"
            />
          )}
        </View>

        <View style={styles.detectionsContainer}>
          <ThemedText type="subtitle" style={styles.detectionsTitle}>
            Detected Objects
          </ThemedText>
          <ThemedText>
            Detections: {String(detectionsCount ?? '0')}  |  Time: {String(processingTime ?? '')}s
          </ThemedText>
          <ThemedText style={{ marginTop: 4, opacity: 0.7 }}>
            Original: {String(originalWidth ?? '')} × {String(originalHeight ?? '')}
          </ThemedText>
        </View>
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
    padding: 20,
  },
  title: {
    textAlign: 'center',
    marginBottom: 20,
  },
  imageContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    padding: 20,
  },
  imageInfo: {
    fontSize: 12,
    opacity: 0.6,
    marginTop: 4,
  },
  detectionsContainer: {
    minHeight: 100,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    padding: 16,
  },
  detectionsTitle: {
    marginBottom: 12,
  },
  placeholderText: {
    textAlign: 'center',
    opacity: 0.7,
  },
});

