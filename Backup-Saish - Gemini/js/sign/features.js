// Hand Landmark Indices
const LM = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + ((a.z || 0) - (b.z || 0)) ** 2);
}

function angle(lm, a, b, c) {
  const ab = { x: lm[a].x - lm[b].x, y: lm[a].y - lm[b].y };
  const cb = { x: lm[c].x - lm[b].x, y: lm[c].y - lm[b].y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mA = Math.sqrt(ab.x ** 2 + ab.y ** 2);
  const mB = Math.sqrt(cb.x ** 2 + cb.y ** 2);
  if (mA === 0 || mB === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (mA * mB)))) * (180 / Math.PI);
}

export function extractFeatures(lm, handLabel) {
  const isRight = handLabel === 'Right';
  const w = lm[LM.WRIST];

  // Finger extension
  const thumbExt = isRight ? lm[LM.THUMB_TIP].x < lm[LM.THUMB_IP].x : lm[LM.THUMB_TIP].x > lm[LM.THUMB_IP].x;
  const indexExt = lm[LM.INDEX_TIP].y < lm[LM.INDEX_PIP].y;
  const middleExt = lm[LM.MIDDLE_TIP].y < lm[LM.MIDDLE_PIP].y;
  const ringExt = lm[LM.RING_TIP].y < lm[LM.RING_PIP].y;
  const pinkyExt = lm[LM.PINKY_TIP].y < lm[LM.PINKY_PIP].y;
  const fingers = [thumbExt, indexExt, middleExt, ringExt, pinkyExt];
  const extCount = fingers.filter(Boolean).length;

  // Curl angles
  const curls = [
    angle(lm, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP),
    angle(lm, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP),
    angle(lm, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
    angle(lm, LM.RING_MCP, LM.RING_PIP, LM.RING_TIP),
    angle(lm, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP),
  ];

  // Key distances
  const thumbIndexDist = dist(lm[LM.THUMB_TIP], lm[LM.INDEX_TIP]);
  const thumbMiddleDist = dist(lm[LM.THUMB_TIP], lm[LM.MIDDLE_TIP]);
  const thumbRingDist = dist(lm[LM.THUMB_TIP], lm[LM.RING_TIP]);
  const thumbPinkyDist = dist(lm[LM.THUMB_TIP], lm[LM.PINKY_TIP]);
  const indexMiddleDist = dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP]);
  const indexPinkyDist = dist(lm[LM.INDEX_TIP], lm[LM.PINKY_TIP]);
  const thumbIndexMcpDist = dist(lm[LM.THUMB_TIP], lm[LM.INDEX_MCP]);

  // Thumb position analysis
  const thumbAcross = isRight ? lm[LM.THUMB_TIP].x > lm[LM.MIDDLE_MCP].x : lm[LM.THUMB_TIP].x < lm[LM.MIDDLE_MCP].x;
  const thumbTucked = lm[LM.THUMB_TIP].y > lm[LM.INDEX_MCP].y && dist(lm[LM.THUMB_TIP], lm[LM.INDEX_MCP]) < 0.06;
  const thumbOnSide = Math.abs(lm[LM.THUMB_TIP].y - lm[LM.INDEX_MCP].y) < 0.04 && dist(lm[LM.THUMB_TIP], lm[LM.INDEX_PIP]) < 0.06;
  const thumbOnMiddle = dist(lm[LM.THUMB_TIP], lm[LM.MIDDLE_PIP]) < 0.045;
  const thumbOnRing = dist(lm[LM.THUMB_TIP], lm[LM.RING_PIP]) < 0.045;
  const thumbOnPinky = dist(lm[LM.THUMB_TIP], lm[LM.PINKY_PIP]) < 0.045;
  const thumbIndexTouch = thumbIndexDist < 0.045;

  // Palm direction
  const vI = { x: lm[LM.INDEX_MCP].x - w.x, y: lm[LM.INDEX_MCP].y - w.y };
  const vP = { x: lm[LM.PINKY_MCP].x - w.x, y: lm[LM.PINKY_MCP].y - w.y };
  const crossZ = vI.x * vP.y - vI.y * vP.x;
  const palmFacing = isRight ? crossZ > 0 : crossZ < 0;

  // Hand angle (wrist → middle MCP)
  const handAngle = Math.atan2(lm[LM.MIDDLE_MCP].y - w.y, lm[LM.MIDDLE_MCP].x - w.x) * (180 / Math.PI);

  // Centroid
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < 21; i++) { cx += lm[i].x; cy += lm[i].y; cz += (lm[i].z || 0); }
  const centroid = { x: cx / 21, y: cy / 21, z: cz / 21 };

  // Index-middle spread
  const indexMiddleSpread = indexMiddleDist > 0.05;
  const ringPinkyTouch = dist(lm[LM.RING_TIP], lm[LM.PINKY_TIP]) < 0.035;

  // Index pointing up check
  const indexPointUp = indexExt && !middleExt && !ringExt && !pinkyExt && lm[LM.INDEX_TIP].y < lm[LM.INDEX_MCP].y;

  // Hand bounding box size (for normalization)
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (let i = 0; i < 21; i++) {
    if (lm[i].x < minX) minX = lm[i].x;
    if (lm[i].x > maxX) maxX = lm[i].x;
    if (lm[i].y < minY) minY = lm[i].y;
    if (lm[i].y > maxY) maxY = lm[i].y;
  }
  const handWidth = maxX - minX;
  const handHeight = maxY - minY;

  return {
    isRight, fingers, extCount, curls,
    thumbExt, indexExt, middleExt, ringExt, pinkyExt,
    thumbIndexDist, thumbMiddleDist, thumbRingDist, thumbPinkyDist,
    indexMiddleDist, indexPinkyDist, thumbIndexMcpDist,
    thumbAcross, thumbTucked, thumbOnSide, thumbOnMiddle, thumbOnRing, thumbOnPinky, thumbIndexTouch,
    palmFacing, handAngle, centroid,
    indexMiddleSpread, ringPinkyTouch, indexPointUp,
    handWidth, handHeight,
    wrist: { x: w.x, y: w.y, z: w.z || 0 },
    landmarks: lm,
  };
}

export { LM, dist };
