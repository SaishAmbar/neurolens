// DynamicGestureClassifier — detects motion-based signs using trajectory patterns

export class DynamicGestureClassifier {
  classify(features, motion) {
    if (!motion || !motion.hasMotion) return null;

    const matches = [];
    const pattern = motion.pattern;
    const traj = motion.trajectory;
    const speed = motion.speed;
    const f = features;

    // HELLO — open palm + wave pattern
    if (pattern === 'wave' && f.extCount >= 4) {
      matches.push({ sign: 'HELLO', confidence: 85, type: 'dynamic' });
    }

    // NO — fist or pointing + horizontal shake
    if (pattern === 'wave' && f.extCount <= 2 && speed > 0.03) {
      matches.push({ sign: 'NO', confidence: 80, type: 'dynamic' });
    }

    // YES — fist + vertical nod
    if (pattern === 'nod' && f.extCount <= 1) {
      matches.push({ sign: 'YES', confidence: 78, type: 'dynamic' });
    }

    // STOP — open palm + push forward
    if (pattern === 'push' && f.extCount >= 4 && f.palmFacing) {
      matches.push({ sign: 'STOP', confidence: 82, type: 'dynamic' });
    }

    // COME — beckoning = pull motion + curling fingers
    if (pattern === 'pull' && f.extCount >= 2) {
      matches.push({ sign: 'COME', confidence: 72, type: 'dynamic' });
    }

    // GO — pointing + sweep right
    if ((traj === 'right' || pattern === 'sweep-right') && f.indexExt && !f.middleExt) {
      matches.push({ sign: 'GO', confidence: 75, type: 'dynamic' });
    }

    // THANK YOU — open hand from chin forward
    if ((traj === 'down' || pattern === 'push') && f.palmFacing && f.extCount >= 3) {
      matches.push({ sign: 'THANK YOU', confidence: 70, type: 'dynamic' });
    }

    // ME — pointing at self (index pointing inward)
    if (f.indexPointUp && f.extCount === 1 && (traj === 'still' || !motion.hasMotion)) {
      matches.push({ sign: 'ME', confidence: 68, type: 'dynamic' });
    }

    // YOU — pointing outward
    if (f.indexExt && !f.middleExt && !f.ringExt && !f.pinkyExt && traj === 'still') {
      matches.push({ sign: 'YOU', confidence: 65, type: 'dynamic' });
    }

    // HELP — fist on palm moving up
    if (traj === 'up' && f.extCount <= 1 && speed > 0.04) {
      matches.push({ sign: 'HELP', confidence: 68, type: 'dynamic' });
    }

    // PLEASE — open palm circular on chest
    if (f.palmFacing && f.extCount >= 4 && speed > 0.02 && speed < 0.08) {
      matches.push({ sign: 'PLEASE', confidence: 60, type: 'dynamic' });
    }

    // Sort by confidence
    matches.sort((a, b) => b.confidence - a.confidence);
    return matches.length > 0 ? matches : null;
  }
}
