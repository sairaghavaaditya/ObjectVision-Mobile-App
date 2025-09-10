import { Image } from 'expo-image';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React from 'react';
import { Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function ResultsScreen() {
  const { imageUri, annotatedUri, detectionsCount, processingTime, originalWidth, originalHeight, detectionsJson, backendMessage, clsLabel, clsProb, caption } = useLocalSearchParams();
  const router = useRouter();

  const parsedDetections: Array<{ class_name: string; confidence: number; box: { x1: number; y1: number; x2: number; y2: number } }> = React.useMemo(() => {
    try {
      if (!detectionsJson) return [];
      const arr = JSON.parse(String(detectionsJson));
      return Array.isArray(arr) ? arr : [];
    } catch {
      return [];
    }
  }, [detectionsJson]);

  const handleDownload = () => {
    if (!annotatedUri) return;
    if (Platform.OS === 'web') {
      try {
        const a = document.createElement('a');
        a.href = String(annotatedUri);
        a.download = 'annotated.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
      } catch {}
    }
  };

  const probPct = (() => {
    const probNum = clsProb ? Number(String(clsProb)) : NaN;
    return isNaN(probNum) ? '' : ` (${(probNum * 100).toFixed(1)}%)`;
  })();

  const primaryLabel = (clsLabel && String(clsLabel)) || (parsedDetections[0]?.class_name ? `${parsedDetections[0].class_name}` : undefined) || (caption ? String(caption) : undefined);
  const secondaryMeta = clsLabel ? probPct : (parsedDetections[0]?.confidence ? ` (${(parsedDetections[0].confidence * 100).toFixed(1)}%)` : '');

  return (
    <SafeAreaView style={styles.container}>
      <View className="w-full max-w-5xl mx-auto px-4 py-6 gap-4" style={styles.page}>
        {/* Title */}
        <View className="items-center gap-1" style={styles.headerWrap}>
          <Text className="text-3xl font-extrabold" style={styles.title}>Detection Results</Text>
          {backendMessage ? (
            <Text className="text-sm opacity-70" style={styles.subtle}>{String(backendMessage)}</Text>
          ) : null}
        </View>

        {/* Image Card */}
        <View className="rounded-2xl bg-neutral-100 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 overflow-hidden" style={styles.card}>
          <Image
            source={{ uri: String(annotatedUri || imageUri) }}
            style={styles.image}
            contentFit="contain"
          />
        </View>

        {/* Main Caption below image */}
        <View className="items-center" style={{ alignItems: 'center' }}>
          <Text style={styles.captionText}>
            {primaryLabel ? `This looks like: ${primaryLabel}${secondaryMeta}` : 'No description available.'}
          </Text>
        </View>

        {/* Stats + Actions */}
        <View className="rounded-2xl border border-neutral-200 dark:border-neutral-700 bg-white/80 dark:bg-neutral-900/60 p-4 gap-3" style={styles.card}>
          <View className="flex-row flex-wrap gap-8" style={styles.statsRow}>
            <Text style={styles.stat}>Detections: {String(detectionsCount ?? parsedDetections.length)}</Text>
            <Text style={styles.stat}>Time: {String(processingTime ?? '')}s</Text>
            <Text style={styles.stat}>Original: {String(originalWidth ?? '')} × {String(originalHeight ?? '')}</Text>
          </View>

          <View className="flex-row gap-3" style={styles.actionsRow}>
            <TouchableOpacity onPress={() => router.back()} activeOpacity={0.85} className="flex-1 items-center justify-center rounded-xl bg-primary border border-primary px-4 py-3" style={[styles.btn, styles.btnPrimary]}>
              <Text className="text-white font-semibold" style={styles.btnPrimaryText}>Back</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => router.replace('/')} activeOpacity={0.85} className="flex-1 items-center justify-center rounded-xl border border-neutral-300 px-4 py-3" style={[styles.btn, styles.btnOutline]}>
              <Text className="font-semibold" style={styles.btnOutlineText}>New Image</Text>
            </TouchableOpacity>
            {Platform.OS === 'web' ? (
              <TouchableOpacity onPress={handleDownload} activeOpacity={0.85} className="flex-1 items-center justify-center rounded-xl border border-neutral-300 px-4 py-3" style={[styles.btn, styles.btnOutline]}>
                <Text className="font-semibold" style={styles.btnOutlineText}>Download</Text>
              </TouchableOpacity>
            ) : null}
          </View>
        </View>

        {/* Detected objects list */}
        <View className="rounded-2xl border border-neutral-200 dark:border-neutral-700 bg-white/80 dark:bg-neutral-900/60 p-4" style={styles.card}>
          <Text className="text-lg font-bold mb-3" style={styles.sectionTitle}>Detected Objects</Text>
          {parsedDetections.length > 0 ? (
            <View className="gap-2" style={{ gap: 8 }}>
              {parsedDetections.map((d, idx) => (
                <View key={idx} className="flex-row items-center justify-between rounded-xl border border-neutral-200 dark:border-neutral-700 px-3 py-2" style={styles.row}>
                  <Text className="font-medium" style={styles.rowText}>{idx + 1}. {d.class_name}</Text>
                  <Text className="opacity-80" style={styles.rowMeta}>{(d.confidence * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>
          ) : (
            <Text className="opacity-70" style={styles.subtle}>No detections above threshold.</Text>
          )}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  page: { maxWidth: 1100, alignSelf: 'center' },
  headerWrap: { alignItems: 'center' },
  title: { fontSize: 28, fontWeight: '800', textAlign: 'center' },
  subtle: { opacity: 0.7, fontSize: 13 },
  card: { borderRadius: 16, borderWidth: 1, borderColor: '#E5E7EB', backgroundColor: 'rgba(255,255,255,0.9)', padding: 12 },
  image: { width: '100%', height: 420, borderRadius: 12 },
  captionText: { marginTop: 4, fontSize: 16, fontWeight: '600' },
  statsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 16 },
  stat: { fontWeight: '600' },
  actionsRow: { flexDirection: 'row', gap: 12 },
  btn: { flex: 1, borderRadius: 12, paddingVertical: 12, alignItems: 'center', justifyContent: 'center' },
  btnPrimary: { backgroundColor: '#20B2AA', borderWidth: 1, borderColor: '#20B2AA' },
  btnPrimaryText: { color: '#fff', fontWeight: '700' },
  btnOutline: { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#D1D5DB' },
  btnOutlineText: { color: '#111827', fontWeight: '700' },
  sectionTitle: { fontSize: 18, fontWeight: '700', marginBottom: 8 },
  row: { borderRadius: 12, borderWidth: 1, borderColor: '#E5E7EB', paddingHorizontal: 12, paddingVertical: 8, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  rowText: { fontWeight: '600' },
  rowMeta: { opacity: 0.8 },
});

