# EntityState KV Store Specification

**For Web UI Integration**

## Overview

Each ISR entity publishes to a **single consolidated KV key** containing all entity state and subsignals. Multiple data streams (detections, analytics, mission intel) update different scopes within the same EntityState object.

---

## KV Key Format

```
Key: {entity_id}
Example: 1048bff5-5b97-4fa8-a0f1-061662b32163
```

**One key per entity** - all state consolidated into a single value.

---

## EntityState Structure

```json
{
  "entity_id": "1048bff5-5b97-4fa8-a0f1-061662b32163",
  "org_id": "2f160b52-37fa-4746-882e-e08ffc395e16",
  "device_id": "b546cd5c6dc0b878",
  "entity_type": "isr_sensor",
  "status": "active",
  "is_live": true,
  "updated_at": "2025-11-21T14:30:45.123Z",

  "mission": {
    "mode": "yoloe_c4isr",
    "status": "operational",
    "started_at": "2025-11-21T14:00:00.000Z"
  },

  "detections": {
    "timestamp": "2025-11-21T14:30:45.123Z",
    "objects": {
      "track_abc123": {
        "track_id": "track_abc123",
        "label": "person",
        "confidence": 0.95,
        "first_seen": "2025-11-21T14:28:12.456Z",
        "last_seen": "2025-11-21T14:30:45.123Z",
        "frame_count": 142,
        "avg_confidence": 0.93,
        "is_active": true,
        "threat_level": "LOW_THREAT",
        "suspicious_indicators": [],
        "area": 0.25,
        "current_bbox": {
          "x_min": 0.195,
          "y_min": 0.077,
          "x_max": 0.870,
          "y_max": 0.997
        }
      }
    }
  },

  "analytics": {
    "timestamp": "2025-11-21T14:30:45.123Z",
    "summary": {
      "total_unique_objects": 42,
      "active_objects_count": 5,
      "total_frames_processed": 1250
    },
    "c4isr_summary": {
      "timestamp": "2025-11-21T14:30:45.123Z",
      "active_threat_count": 2,
      "threat_distribution": {
        "LOW_THREAT": 3,
        "MODERATE_THREAT": 1,
        "HIGH_THREAT": 1
      }
    }
  },

  "c4isr": {
    "timestamp": "2025-11-21T14:30:45.123Z",
    "mission": "C4ISR",
    "threat_intelligence": {
      "threat_summary": {
        "total_threats": 2,
        "alert_level": "NORMAL",
        "threat_distribution": {
          "LOW_THREAT": 3,
          "MODERATE_THREAT": 1,
          "HIGH_THREAT": 1
        }
      },
      "threat_alerts": [
        {
          "track_id": "track_xyz789",
          "alert_type": "HIGH_THREAT",
          "reason": "Weapon detected with high confidence",
          "timestamp": "2025-11-21T14:29:30.456Z"
        }
      ]
    }
  }
}
```

---

## Subsignal Scopes & Update Patterns

### 1. **`detections`** - Object Tracking
**Updated:** Every frame with persistent objects (typically 1-30 Hz depending on detection rate)

**Scope:**
- `objects` - Dictionary of currently tracked objects keyed by track_id
- Each object includes tracking metadata, confidence, threat level, bounding box

**UI Use Cases:**
- Real-time object overlay on video feed
- Track count and object list
- Object persistence visualization

---

### 2. **`analytics`** - Performance Metrics
**Updated:** Every frame with persistent objects (same frequency as detections)

**Scope:**
- `summary` - Overall detection statistics
  - Total unique objects seen
  - Active objects count
  - Frames processed
- `c4isr_summary` - Threat-specific analytics (C4ISR mode only)
  - Active threat count
  - Threat distribution by level

**UI Use Cases:**
- Dashboard metrics (total objects, active count)
- Performance graphs (frames/sec)
- Threat level distribution charts

---

### 3. **`c4isr`** - Mission Intelligence
**Updated:** Every frame in C4ISR detection mode (when threats detected)

**Scope:**
- `mission` - Mission type identifier ("C4ISR")
- `threat_intelligence` - Threat detection results
  - `threat_summary` - Aggregate threat metrics
  - `threat_alerts` - Array of active threat alerts with details

**UI Use Cases:**
- Threat alert panel
- Mission status indicator
- Alert history timeline
- High-priority threat notifications

---

## Update Frequency

| Subsignal | Update Rate | Trigger |
|-----------|-------------|---------|
| `detections` | 1-30 Hz | When persistent objects present |
| `analytics` | 1-30 Hz | With every detection update |
| `c4isr` | 1-30 Hz | C4ISR mode only, when threats exist |
| Top-level `updated_at` | 1-30 Hz | On any subsignal update |

**Note:** Updates are throttled by "smart publishing" - only publishes when significant state changes occur (movement, confidence change, new/lost objects).

---

## Web UI Integration Guide

### Subscribing to Updates

```javascript
// NATS JetStream KV Watch
const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');

// Watch specific entity
const watch = await kv.watch({
  key: entityId  // e.g., "1048bff5-5b97-4fa8-a0f1-061662b32163"
});

for await (const entry of watch) {
  const entityState = JSON.parse(entry.string());
  updateUI(entityState);
}
```

### Reading Current State

```javascript
// Get latest state for entity
const entry = await kv.get(entityId);
const entityState = JSON.parse(entry.string());
```

### Recommended UI Components

**1. Entity Status Card**
```javascript
{
  entity_id: entityState.entity_id,
  status: entityState.status,
  is_live: entityState.is_live,
  last_update: entityState.updated_at,
  active_objects: entityState.analytics.summary.active_objects_count
}
```

**2. Object Detection Panel**
```javascript
Object.values(entityState.detections.objects)
  .filter(obj => obj.is_active)
  .map(obj => ({
    id: obj.track_id,
    label: obj.label,
    confidence: obj.confidence,
    threat: obj.threat_level,
    bbox: obj.current_bbox
  }))
```

**3. Threat Alert List**
```javascript
entityState.c4isr?.threat_intelligence?.threat_alerts || []
```

**4. Analytics Dashboard**
```javascript
{
  totalObjects: entityState.analytics.summary.total_unique_objects,
  activeCount: entityState.analytics.summary.active_objects_count,
  framesProcessed: entityState.analytics.summary.total_frames_processed,
  threatDistribution: entityState.analytics.c4isr_summary?.threat_distribution
}
```

---

## Key Takeaways

✅ **Single Key per Entity** - No fragmented data, one KV get retrieves complete state

✅ **Subsignal Isolation** - Different data streams update their own scopes without conflicts

✅ **Real-time Updates** - High-frequency updates (1-30 Hz) with smart throttling

✅ **Consistent Structure** - All entities follow the same schema

✅ **Mission-Specific Data** - C4ISR fields only present when in threat detection mode

---

## Field Guarantees

### Always Present
- `entity_id`, `org_id`, `device_id`, `entity_type`
- `status`, `is_live`, `updated_at`
- `mission` - MissionState object with mode, status, started_at
- `detections`, `analytics` (may be empty `{}`)

### Conditionally Present
- `c4isr` - Only in C4ISR detection mode
- `analytics.c4isr_summary` - Only when C4ISR active
- `detections.objects` - Empty when no tracked objects

### Timestamps
- All timestamps are ISO8601 UTC format
- Top-level `updated_at` reflects last subsignal update
- Each subsignal has its own `timestamp`

---

## Example Queries

**Get all active entities:**
```bash
nats kv ls CONSTELLATION_GLOBAL_STATE
```

**Get specific entity state:**
```bash
nats kv get CONSTELLATION_GLOBAL_STATE {entity_id}
```

**Watch for updates:**
```bash
nats kv watch CONSTELLATION_GLOBAL_STATE --key={entity_id}
```

---

## Version

**Spec Version:** 1.0.0
**Last Updated:** 2025-11-21
**ISR Client Version:** 1.0.0
