# Constellation Overwatch: Downstream Data Pipeline PRD

**Product Requirements Document**
**Version:** 1.0.0
**Last Updated:** 2025-11-21
**Audience:** Web UI Developers, Backend Engineers, Data Pipeline Teams

---

## Table of Contents

1. [Overview](#overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Event Stream: Detection Events](#event-stream-detection-events)
4. [Key-Value Store: EntityState](#key-value-store-entitystate)
5. [Integration Patterns](#integration-patterns)
6. [Implementation Guide](#implementation-guide)
7. [Example Queries](#example-queries)

---

## Overview

The Constellation Overwatch ISR system publishes data through two complementary channels:

1. **JetStream Events** - Real-time detection events (high-frequency, ephemeral)
2. **KV Store** - Consolidated entity state (current state, queryable)

This dual-channel architecture provides:
- ✅ Real-time event streaming for immediate response
- ✅ State snapshots for dashboards and queries
- ✅ Historical replay via JetStream retention
- ✅ Efficient state queries without event processing

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ISR Detection Client                      │
│                                                              │
│  ┌────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Detector  │───▶│   Tracking   │───▶│ Communication │  │
│  │  (YOLO/    │    │   Service    │    │   Service     │  │
│  │   SAM/etc) │    │              │    │               │  │
│  └────────────┘    └──────────────┘    └───────────────┘  │
│                                                │             │
└────────────────────────────────────────────────┼─────────────┘
                                                 │
                    ┌────────────────────────────┴───────────────────┐
                    │                                                │
                    ▼                                                ▼
         ┌──────────────────────┐                      ┌──────────────────────┐
         │  NATS JetStream      │                      │   NATS KV Store      │
         │  CONSTELLATION_      │                      │   CONSTELLATION_     │
         │  EVENTS              │                      │   GLOBAL_STATE       │
         │                      │                      │                      │
         │  Subject:            │                      │  Key: {entity_id}    │
         │  constellation.      │                      │                      │
         │  events.isr.         │                      │  Value: EntityState  │
         │  {org}.{entity}      │                      │  (consolidated)      │
         └──────────────────────┘                      └──────────────────────┘
                    │                                                │
                    │                                                │
         ┌──────────┴────────────┐                      ┌───────────┴──────────┐
         │                       │                      │                      │
         ▼                       ▼                      ▼                      ▼
    ┌─────────┐           ┌──────────┐          ┌──────────┐         ┌──────────┐
    │ Event   │           │ Backend  │          │ Web UI   │         │ Backend  │
    │ Logger  │           │ Service  │          │ Dashboard│         │ API      │
    │         │           │ (Stream) │          │ (Watch)  │         │ (Query)  │
    └─────────┘           └──────────┘          └──────────┘         └──────────┘
```

### Key Principles

1. **Events for Reactions** - Subscribe to detection events for real-time alerts
2. **KV for State** - Query KV store for current entity state and dashboards
3. **Single Source of Truth** - Entity state in KV is authoritative
4. **Smart Publishing** - Only publish on significant state changes

---

## Event Stream: Detection Events

### Purpose

Real-time notification of object detections with immediate delivery guarantees.

### NATS Subject Pattern

```
constellation.events.isr.{organization_id}.{entity_id}
```

**Example:**
```
constellation.events.isr.2f160b52-37fa-4746-882e-e08ffc395e16.1048bff5-5b97-4fa8-a0f1-061662b32163
```

### Event Payload Structure

```json
{
  "timestamp": "2025-11-21T14:30:45.123456+00:00",
  "event_type": "detection",
  "entity_id": "1048bff5-5b97-4fa8-a0f1-061662b32163",
  "device_id": "b546cd5c6dc0b878",
  "organization_id": "2f160b52-37fa-4746-882e-e08ffc395e16",
  "detection": {
    "track_id": "clx7y3k2r0000qzrm8n7qh3k1",
    "model_type": "yoloe-c4isr-threat-detection",
    "label": "person",
    "confidence": 0.96,
    "bbox": {
      "x_min": 0.189,
      "y_min": 0.179,
      "x_max": 0.837,
      "y_max": 0.997
    },
    "timestamp": "2025-11-21T14:30:45.123456+00:00",
    "metadata": {
      "native_id": 1,
      "threat_level": "LOW_THREAT",
      "suspicious_indicators": []
    }
  }
}
```

### Top-Level Envelope Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | ISO8601 | ✅ | Event publication time |
| `event_type` | string | ✅ | Always `"detection"` for detection events |
| `entity_id` | string | ✅ | Entity UUID (from .env) |
| `device_id` | string | ✅ | Hardware fingerprint hash |
| `organization_id` | string | ✅ | Organization UUID (from .env) |
| `detection` | object | ✅ | Detection payload (see below) |

### Detection Object Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `track_id` | string | ✅ | Globally unique CUID for tracked object |
| `model_type` | string | ✅ | Detection model identifier |
| `label` | string | ✅ | Object class (person, car, weapon, etc.) |
| `confidence` | float | ✅ | Detection confidence (0.0-1.0) |
| `bbox` | object | ✅ | Normalized bounding box (0-1 range) |
| `timestamp` | ISO8601 | ✅ | Detection timestamp |
| `metadata` | object | ✅ | Model-specific metadata |

### Metadata Fields (Model-Specific)

**YOLOE C4ISR:**
```json
"metadata": {
  "native_id": 1,
  "threat_level": "HIGH_THREAT",  // LOW_THREAT | MODERATE_THREAT | HIGH_THREAT
  "suspicious_indicators": ["high_confidence_weapon_detection"]
}
```

**SAM2 Segmentation:**
```json
"metadata": {
  "native_id": 0,
  "mask": [[0, 1, 1, ...], ...],
  "area": 2547.5
}
```

**RT-DETR:**
```json
"metadata": {
  "native_id": 2,
  "class_id": 0
}
```

### Smart Publishing Triggers

Detection events are published only when:

1. ✅ **New object** - First appearance of track_id
2. ✅ **Significant movement** - Object moves > `SIGINT_MOVEMENT_THRESHOLD` (default: 5%)
3. ✅ **Confidence change** - Confidence changes > `SIGINT_CONFIDENCE_THRESHOLD` (default: 10%)
4. ✅ **Label change** - Object classification changes
5. ✅ **Threat level change** - Threat assessment changes (C4ISR mode)

This reduces bandwidth by 70-90% while maintaining data fidelity.

### Event Frequency

- **Typical:** 1-30 Hz per entity (throttled by smart publishing)
- **Peak:** Up to 60 Hz during high-activity periods
- **Minimum:** 0 Hz when no objects detected

---

## Key-Value Store: EntityState

### Purpose

Consolidated entity state snapshot for efficient queries and dashboards.

### KV Key Format

```
Key: {entity_id}
Example: 1048bff5-5b97-4fa8-a0f1-061662b32163
```

**One key per entity** - complete state in single value.

### EntityState Structure

```json
{
  "entity_id": "1048bff5-5b97-4fa8-a0f1-061662b32163",
  "org_id": "2f160b52-37fa-4746-882e-e08ffc395e16",
  "device_id": "b546cd5c6dc0b878",
  "status": "active",
  "is_live": true,
  "updated_at": "2025-11-21T14:30:45.123Z",

  "detections": {
    "timestamp": "2025-11-21T14:30:45.123Z",
    "objects": {
      "clx7y3k2r0000qzrm8n7qh3k1": {
        "track_id": "clx7y3k2r0000qzrm8n7qh3k1",
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
          "track_id": "clx7y3k2r0000qzrm8n7qh3k1",
          "alert_type": "HIGH_THREAT",
          "reason": "Weapon detected with high confidence",
          "timestamp": "2025-11-21T14:29:30.456Z"
        }
      ]
    }
  }
}
```

### Top-Level Fields

| Field | Type | Always Present | Description |
|-------|------|----------------|-------------|
| `entity_id` | string | ✅ | Entity UUID |
| `org_id` | string | ✅ | Organization UUID (shorter form for KV) |
| `device_id` | string | ✅ | Hardware fingerprint |
| `status` | string | ✅ | Entity status (`active`, `offline`, `error`) |
| `is_live` | boolean | ✅ | Real-time connection status |
| `updated_at` | ISO8601 | ✅ | Last update timestamp |

### Subsignal Scopes

#### 1. `detections` - Object Tracking

**Update Rate:** 1-30 Hz (when persistent objects exist)

**Purpose:** Current tracked objects with metadata

**Structure:**
```json
{
  "timestamp": "...",
  "objects": {
    "{track_id}": {
      "track_id": "...",
      "label": "...",
      "confidence": 0.95,
      "first_seen": "...",
      "last_seen": "...",
      "frame_count": 142,
      "avg_confidence": 0.93,
      "is_active": true,
      "threat_level": "LOW_THREAT",
      "current_bbox": {...}
    }
  }
}
```

**UI Use Cases:**
- Real-time object overlay on video
- Active object count
- Track persistence visualization

---

#### 2. `analytics` - Performance Metrics

**Update Rate:** 1-30 Hz (with every detection update)

**Purpose:** Aggregate statistics and performance data

**Structure:**
```json
{
  "timestamp": "...",
  "summary": {
    "total_unique_objects": 42,
    "active_objects_count": 5,
    "total_frames_processed": 1250
  },
  "c4isr_summary": {  // C4ISR mode only
    "active_threat_count": 2,
    "threat_distribution": {
      "LOW_THREAT": 3,
      "MODERATE_THREAT": 1,
      "HIGH_THREAT": 1
    }
  }
}
```

**UI Use Cases:**
- Dashboard metrics
- Performance graphs (FPS, throughput)
- Threat level distribution charts

---

#### 3. `c4isr` - Mission Intelligence

**Update Rate:** 1-30 Hz (C4ISR mode only, when threats exist)

**Purpose:** Threat detection and mission-specific intelligence

**Structure:**
```json
{
  "timestamp": "...",
  "mission": "C4ISR",
  "threat_intelligence": {
    "threat_summary": {
      "total_threats": 2,
      "alert_level": "NORMAL",  // NORMAL | ELEVATED | HIGH
      "threat_distribution": {...}
    },
    "threat_alerts": [
      {
        "track_id": "...",
        "alert_type": "HIGH_THREAT",
        "reason": "Weapon detected with high confidence",
        "timestamp": "..."
      }
    ]
  }
}
```

**UI Use Cases:**
- Threat alert panel
- Mission status indicator
- Alert history timeline
- High-priority notifications

---

### Field Guarantees

#### Always Present
- `entity_id`, `org_id`, `device_id`
- `status`, `is_live`, `updated_at`
- `detections`, `analytics` (may be empty `{}`)

#### Conditionally Present
- `c4isr` - Only in C4ISR detection mode
- `analytics.c4isr_summary` - Only when C4ISR active
- `detections.objects` - Empty `{}` when no tracked objects

#### Timestamps
- All timestamps are ISO8601 UTC format
- Top-level `updated_at` reflects last subsignal update
- Each subsignal has its own `timestamp`

---

## Integration Patterns

### Pattern 1: Real-Time Event Processing

**Use Case:** Immediate alerts, logging, event-driven workflows

```javascript
// Subscribe to detection events
const js = nc.jetstream();
const consumer = await js.consumers.get('CONSTELLATION_EVENTS', 'detection-processor');

for await (const msg of consumer.consume()) {
  const event = JSON.parse(msg.string());

  if (event.event_type === 'detection') {
    const detection = event.detection;

    // Process high-threat detections immediately
    if (detection.metadata.threat_level === 'HIGH_THREAT') {
      await sendAlert(event.entity_id, detection);
    }
  }

  msg.ack();
}
```

**Benefits:**
- ✅ Real-time processing
- ✅ Event replay capability
- ✅ Guaranteed delivery
- ✅ Backpressure handling

---

### Pattern 2: Current State Queries

**Use Case:** Dashboards, API endpoints, status pages

```javascript
// Get current entity state
const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');
const entry = await kv.get(entityId);
const entityState = JSON.parse(entry.string());

// Query specific data
const activeObjects = Object.values(entityState.detections.objects)
  .filter(obj => obj.is_active);

const threatLevel = entityState.c4isr?.threat_intelligence?.threat_summary?.alert_level;
const activeCount = entityState.analytics.summary.active_objects_count;
```

**Benefits:**
- ✅ Single query for complete state
- ✅ No event processing required
- ✅ Low latency
- ✅ Efficient for dashboards

---

### Pattern 3: Hybrid - Watch State Changes

**Use Case:** Real-time dashboards, live monitoring

```javascript
// Watch for KV state changes
const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');
const watch = await kv.watch({ key: entityId });

for await (const entry of watch) {
  const entityState = JSON.parse(entry.string());

  // Update UI components
  updateEntityCard(entityState);
  updateObjectList(entityState.detections.objects);
  updateThreatPanel(entityState.c4isr?.threat_intelligence);
  updateMetrics(entityState.analytics);
}
```

**Benefits:**
- ✅ Real-time UI updates
- ✅ Complete state on each update
- ✅ Efficient (only changed states)
- ✅ No event aggregation logic needed

---

### Pattern 4: Historical Event Replay

**Use Case:** Forensics, debugging, audit trails

```javascript
// Replay events from specific time
const js = nc.jetstream();
const consumer = await js.consumers.get('CONSTELLATION_EVENTS', {
  deliver_policy: DeliverPolicy.ByStartTime,
  opt_start_time: new Date('2025-11-21T14:00:00Z')
});

for await (const msg of consumer.consume()) {
  const event = JSON.parse(msg.string());

  // Reconstruct timeline
  if (event.event_type === 'detection') {
    await logHistoricalDetection(event);
  }

  msg.ack();
}
```

**Benefits:**
- ✅ Complete event history
- ✅ Time-based queries
- ✅ Audit compliance
- ✅ Debugging capabilities

---

## Implementation Guide

### Web UI Integration

#### 1. Entity Status Card

```javascript
function EntityStatusCard({ entityId }) {
  const [state, setState] = useState(null);

  useEffect(() => {
    const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');
    const watch = await kv.watch({ key: entityId });

    for await (const entry of watch) {
      setState(JSON.parse(entry.string()));
    }
  }, [entityId]);

  return (
    <Card>
      <Status color={state.is_live ? 'green' : 'gray'}>
        {state.status}
      </Status>
      <Metric label="Active Objects">
        {state.analytics.summary.active_objects_count}
      </Metric>
      <Timestamp>{state.updated_at}</Timestamp>
    </Card>
  );
}
```

---

#### 2. Object Detection Panel

```javascript
function ObjectDetectionPanel({ entityId }) {
  const [objects, setObjects] = useState([]);

  useEffect(() => {
    const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');
    const watch = await kv.watch({ key: entityId });

    for await (const entry of watch) {
      const state = JSON.parse(entry.string());
      const activeObjects = Object.values(state.detections.objects || {})
        .filter(obj => obj.is_active);
      setObjects(activeObjects);
    }
  }, [entityId]);

  return (
    <Panel>
      {objects.map(obj => (
        <ObjectCard key={obj.track_id}>
          <Label>{obj.label}</Label>
          <Confidence>{(obj.confidence * 100).toFixed(1)}%</Confidence>
          <ThreatBadge level={obj.threat_level} />
          <BoundingBox bbox={obj.current_bbox} />
        </ObjectCard>
      ))}
    </Panel>
  );
}
```

---

#### 3. Threat Alert List

```javascript
function ThreatAlertPanel({ entityId }) {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');
    const watch = await kv.watch({ key: entityId });

    for await (const entry of watch) {
      const state = JSON.parse(entry.string());
      setAlerts(state.c4isr?.threat_intelligence?.threat_alerts || []);
    }
  }, [entityId]);

  return (
    <AlertList>
      {alerts.map(alert => (
        <Alert key={alert.track_id} severity={alert.alert_type}>
          <AlertIcon type={alert.alert_type} />
          <AlertText>{alert.reason}</AlertText>
          <Timestamp>{alert.timestamp}</Timestamp>
        </Alert>
      ))}
    </AlertList>
  );
}
```

---

### Backend API Integration

#### REST API Endpoint

```python
from fastapi import FastAPI, HTTPException
import nats

app = FastAPI()

@app.get("/api/entities/{entity_id}")
async def get_entity_state(entity_id: str):
    """Get current entity state from KV store."""
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()
    kv = await js.key_value("CONSTELLATION_GLOBAL_STATE")

    try:
        entry = await kv.get(entity_id)
        return json.loads(entry.value.decode())
    except:
        raise HTTPException(status_code=404, detail="Entity not found")

@app.get("/api/entities")
async def list_entities():
    """List all active entities."""
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()
    kv = await js.key_value("CONSTELLATION_GLOBAL_STATE")

    keys = await kv.keys()
    entities = []

    for key in keys:
        entry = await kv.get(key)
        state = json.loads(entry.value.decode())
        if state['is_live']:
            entities.append({
                'entity_id': state['entity_id'],
                'status': state['status'],
                'active_objects': state['analytics']['summary']['active_objects_count']
            })

    return entities
```

---

### Event Stream Consumer

```python
import asyncio
import nats

async def consume_detection_events():
    """Subscribe to detection events for processing."""
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()

    # Durable consumer for reliable processing
    psub = await js.pull_subscribe(
        "constellation.events.isr.>",
        "detection-processor"
    )

    while True:
        msgs = await psub.fetch(batch=10, timeout=1.0)

        for msg in msgs:
            event = json.loads(msg.data.decode())

            if event['event_type'] == 'detection':
                await process_detection(event)

            await msg.ack()

async def process_detection(event):
    """Process individual detection event."""
    detection = event['detection']

    # Log to database
    await db.insert_detection(
        entity_id=event['entity_id'],
        track_id=detection['track_id'],
        label=detection['label'],
        confidence=detection['confidence'],
        threat_level=detection['metadata'].get('threat_level'),
        timestamp=detection['timestamp']
    )

    # Send alerts for high-threat detections
    if detection['metadata'].get('threat_level') == 'HIGH_THREAT':
        await alert_service.send_alert(event)
```

---

## Example Queries

### NATS CLI

```bash
# List all entities in KV store
nats kv ls CONSTELLATION_GLOBAL_STATE

# Get specific entity state
nats kv get CONSTELLATION_GLOBAL_STATE 1048bff5-5b97-4fa8-a0f1-061662b32163

# Watch for entity state changes
nats kv watch CONSTELLATION_GLOBAL_STATE --key=1048bff5-5b97-4fa8-a0f1-061662b32163

# Subscribe to detection events for all entities
nats sub "constellation.events.isr.>"

# Subscribe to specific entity events
nats sub "constellation.events.isr.*.1048bff5-5b97-4fa8-a0f1-061662b32163"
```

---

### JavaScript/TypeScript

```typescript
import { connect, JSONCodec } from 'nats';

const nc = await connect({ servers: 'nats://localhost:4222' });
const js = nc.jetstream();
const jc = JSONCodec();

// Get KV store
const kv = await js.views.kv('CONSTELLATION_GLOBAL_STATE');

// Get entity state
const entry = await kv.get(entityId);
const state = jc.decode(entry.value);

// Watch for updates
const watch = await kv.watch({ key: entityId });
for await (const entry of watch) {
  const state = jc.decode(entry.value);
  console.log('Updated state:', state);
}

// Subscribe to events
const sub = nc.subscribe('constellation.events.isr.>');
for await (const msg of sub) {
  const event = jc.decode(msg.data);
  console.log('Detection event:', event);
}
```

---

### Python

```python
import nats
import json

async def main():
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()

    # Get KV store
    kv = await js.key_value("CONSTELLATION_GLOBAL_STATE")

    # Get entity state
    entry = await kv.get(entity_id)
    state = json.loads(entry.value.decode())

    # Watch for updates
    watcher = await kv.watch(keys=[entity_id])
    async for entry in watcher:
        state = json.loads(entry.value.decode())
        print(f"Updated state: {state}")

    # Subscribe to events
    async def message_handler(msg):
        event = json.loads(msg.data.decode())
        print(f"Detection event: {event}")

    await nc.subscribe("constellation.events.isr.>", cb=message_handler)
```

---

## Performance Characteristics

### Event Stream

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 1,000-10,000 msg/sec | Per stream |
| Latency | < 5ms | Publish to delivery |
| Retention | 24 hours | Configurable |
| Message Size | < 1MB | Typical: 1-5KB |

### KV Store

| Metric | Value | Notes |
|--------|-------|-------|
| Read Latency | < 1ms | Local cache |
| Write Latency | < 5ms | With replication |
| History | 10 revisions | Configurable |
| TTL | 24 hours | Auto-cleanup |
| Max Value Size | 1MB | Per key |

---

## Best Practices

### ✅ DO

1. **Use KV for Dashboards** - Query current state, don't aggregate events
2. **Use Events for Actions** - React to real-time detections
3. **Cache Entity State** - Reduce KV reads with client-side caching
4. **Handle Disconnects** - Gracefully reconnect and resync state
5. **Validate Schemas** - Check for required fields before processing

### ❌ DON'T

1. **Don't Aggregate Events for State** - Use KV store instead
2. **Don't Poll KV** - Use watch for real-time updates
3. **Don't Store Events in DB Without Reason** - JetStream provides retention
4. **Don't Assume All Fields Present** - Check for optional/conditional fields
5. **Don't Ignore Timestamps** - Use them for ordering and staleness detection

---

## Troubleshooting

### Entity State Not Updating

**Symptom:** KV entry exists but doesn't update

**Causes:**
- Client disconnected
- Smart publishing threshold not met
- No persistent objects detected

**Solution:** Check `is_live` field and `updated_at` timestamp

---

### Events Not Received

**Symptom:** JetStream subscription not receiving messages

**Causes:**
- Incorrect subject pattern
- Consumer not configured correctly
- Stream retention policy expired

**Solution:** Verify subject pattern and consumer configuration

---

### Missing Fields in EntityState

**Symptom:** Expected field is `null` or missing

**Causes:**
- Conditional field (e.g., `c4isr` only in C4ISR mode)
- No objects detected yet (`detections.objects` empty)

**Solution:** Check field guarantees section, use optional chaining

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-21 | Initial consolidated PRD |

---

## References

- **ISR Client Repository:** [obj-detection-client](/)
- **NATS Documentation:** https://docs.nats.io
- **JetStream Guide:** https://docs.nats.io/nats-concepts/jetstream
- **KV Store Guide:** https://docs.nats.io/nats-concepts/jetstream/key-value-store
- **Overwatch Backend:** (Contact team for access)

---

## Support

For questions or issues:
- **Technical Issues:** File issue in repository
- **Integration Support:** Contact backend team
- **Schema Questions:** Reference this document

---

**End of Document**
