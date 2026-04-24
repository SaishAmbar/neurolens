// StaticGestureClassifier v2 — Controlled Vocabulary (25 whole-word signs)
// Focuses on HIGH ACCURACY for a curated sign set instead of A-Z letters
// Each sign uses 8-12 weighted feature checks for robust discrimination

// Controlled vocabulary — these are the ONLY signs we recognize
export const VOCABULARY = [
  'I', 'YOU', 'HELLO', 'THANK YOU', 'PLEASE', 'YES', 'NO', 'STOP',
  'GO', 'COME', 'EAT', 'DRINK', 'WANT', 'NEED', 'HELP',
  'HOME', 'SCHOOL', 'FOOD', 'WATER', 'GOOD', 'BAD',
  'SORRY', 'LOVE', 'FRIEND', 'FAMILY',
  'TODAY', 'TOMORROW', 'HAPPY', 'SAD', 'MORE',
];

export class StaticGestureClassifier {
  classify(f, motion) {
    const m = [];
    const add = (sign, score, max) => {
      const c = Math.round((score / max) * 100);
      if (c >= 50) m.push({ sign, confidence: Math.min(c, 98) });
    };

    const mo = motion || {};
    const pat = mo.pattern || 'none';
    const traj = mo.trajectory || 'still';
    const spd = mo.speed || 0;
    const moving = mo.hasMotion || false;

    // ═══ PRONOUNS ═══

    // I / ME — index finger pointing at chest (index up, others down, hand near body center)
    { let s = 0, x = 8;
      if (f.indexExt) s += 2;
      if (!f.middleExt) s++; if (!f.ringExt) s++; if (!f.pinkyExt) s++;
      if (f.extCount === 1 || (f.extCount === 2 && f.thumbExt)) s++;
      // Pointing inward (toward body center — centroid.x near 0.5)
      if (f.centroid.x > 0.35 && f.centroid.x < 0.65) s++;
      if (f.centroid.y > 0.3 && f.centroid.y < 0.7) s++;
      add('I', s, x);
    }

    // YOU — index finger pointing outward
    { let s = 0, x = 8;
      if (f.indexExt) s += 2;
      if (!f.middleExt) s++; if (!f.ringExt) s++; if (!f.pinkyExt) s++;
      if (f.extCount <= 2) s++;
      // Pointing away from body (hand more to the side or forward)
      if (!f.indexPointUp) s++;
      if (!moving || spd < 0.03) s++;
      add('YOU', s, x);
    }

    // ═══ GREETINGS & SOCIAL ═══

    // HELLO — open palm + wave motion
    { let s = 0, x = 10;
      if (f.extCount >= 4) s += 2;
      if (f.palmFacing) s += 2;
      if (pat === 'wave') s += 3;
      else if (moving && spd > 0.02) s++;
      if (f.centroid.y < 0.5) s++; // hand raised
      if (f.thumbExt) s++;
      if (f.indexExt && f.middleExt && f.ringExt && f.pinkyExt) s++;
      add('HELLO', s, x);
    }

    // THANK YOU — flat hand from chin moving forward/down
    { let s = 0, x = 9;
      if (f.extCount >= 3) s++;
      if (f.palmFacing) s++;
      if (traj === 'down' || pat === 'push') s += 2;
      if (f.centroid.y < 0.5) s++; // starts near face
      if (f.indexExt && f.middleExt) s++;
      if (moving && spd > 0.015) s += 2;
      if (!f.thumbAcross) s++;
      add('THANK YOU', s, x);
    }

    // PLEASE — open palm circular on chest
    { let s = 0, x = 8;
      if (f.extCount >= 4) s += 2;
      if (f.palmFacing) s++;
      if (f.centroid.y > 0.35 && f.centroid.y < 0.65) s++; // chest level
      if (moving && spd > 0.01 && spd < 0.08) s += 2;
      if (f.thumbExt) s++;
      if (pat !== 'wave' && pat !== 'nod') s++;
      add('PLEASE', s, x);
    }

    // SORRY — fist circular on chest
    { let s = 0, x = 8;
      if (f.extCount <= 1) s += 2;
      if (!f.indexExt && !f.middleExt && !f.ringExt && !f.pinkyExt) s += 2;
      if (f.centroid.y > 0.3 && f.centroid.y < 0.65) s++;
      if (moving && spd > 0.01 && spd < 0.06) s += 2;
      if (f.thumbOnSide || f.thumbTucked) s++;
      add('SORRY', s, x);
    }

    // ═══ YES / NO ═══

    // YES — fist nodding (vertical oscillation)
    { let s = 0, x = 9;
      if (f.extCount <= 1) s += 2;
      if (!f.indexExt && !f.middleExt) s++;
      if (pat === 'nod') s += 3;
      else if (traj === 'down' || traj === 'up') s++;
      if (f.thumbOnSide || f.thumbExt) s++;
      if (moving) s++;
      if (f.centroid.y < 0.55) s++;
      add('YES', s, x);
    }

    // NO — index+middle wag/snap or horizontal shake
    { let s = 0, x = 9;
      if (f.indexExt && f.middleExt) s += 2;
      if (!f.ringExt && !f.pinkyExt) s++;
      if (pat === 'wave' && f.extCount <= 3) s += 3;
      else if (traj === 'left' || traj === 'right') s++;
      if (f.thumbExt || f.thumbOnSide) s++;
      if (moving && spd > 0.02) s++;
      add('NO', s, x);
    }

    // ═══ ACTIONS ═══

    // STOP — open palm push forward
    { let s = 0, x = 9;
      if (f.extCount >= 4) s += 2;
      if (f.palmFacing) s += 2;
      if (pat === 'push' || traj === 'up') s += 2;
      if (f.centroid.y < 0.5) s++;
      if (!f.thumbAcross) s++;
      if (f.indexExt && f.middleExt && f.ringExt && f.pinkyExt) s++;
      add('STOP', s, x);
    }

    // GO — pointing outward + sweep
    { let s = 0, x = 9;
      if (f.indexExt) s += 2;
      if (!f.middleExt && !f.ringExt && !f.pinkyExt) s++;
      if (traj === 'right' || pat === 'sweep-right') s += 2;
      else if (traj === 'left' || pat === 'sweep-left') s += 2;
      if (moving && spd > 0.02) s += 2;
      if (f.extCount <= 2) s++;
      add('GO', s, x);
    }

    // COME — beckoning motion (pull toward self)
    { let s = 0, x = 9;
      if (f.indexExt) s++;
      if (pat === 'pull') s += 3;
      else if (traj === 'left') s++;
      if (f.palmFacing) s++;
      if (moving && spd > 0.02) s += 2;
      if (f.extCount >= 2 && f.extCount <= 4) s++;
      if (f.centroid.y > 0.3) s++;
      add('COME', s, x);
    }

    // EAT — fingers to mouth repeatedly (closed hand near face, movement)
    { let s = 0, x = 9;
      if (f.extCount <= 2) s++;
      if (f.thumbIndexTouch || f.thumbIndexDist < 0.05) s += 2;
      if (f.centroid.y < 0.45) s += 2; // near face
      if (pat === 'nod' || (moving && spd > 0.01 && spd < 0.06)) s += 2;
      if (f.curls[1] > 60) s++; // fingers somewhat curled
      if (f.thumbExt || f.thumbOnSide) s++;
      add('EAT', s, x);
    }

    // DRINK — C-hand tilting at mouth (thumb+index forming C, near face)
    { let s = 0, x = 9;
      if (f.thumbExt) s++;
      if (f.curls[1] > 50 && f.curls[1] < 140) s++; // C shape
      if (f.thumbIndexDist > 0.03 && f.thumbIndexDist < 0.10) s += 2;
      if (f.centroid.y < 0.45) s += 2; // near face
      if (traj === 'up' || pat === 'nod') s += 2;
      if (f.extCount <= 3) s++;
      add('DRINK', s, x);
    }

    // WANT — open hands pulling toward self (clawing motion)
    { let s = 0, x = 8;
      if (f.extCount >= 3) s++;
      if (f.curls[1] > 40 && f.curls[1] < 130) s++; // semi-curled
      if (pat === 'pull' || traj === 'left') s += 2;
      if (moving) s++;
      if (f.palmFacing) s++;
      if (f.centroid.y > 0.3 && f.centroid.y < 0.6) s++;
      if (f.thumbExt) s++;
      add('WANT', s, x);
    }

    // NEED — index bent, nodding down
    { let s = 0, x = 8;
      if (f.curls[1] > 60 && f.curls[1] < 140) s += 2;
      if (!f.middleExt && !f.ringExt && !f.pinkyExt) s++;
      if (traj === 'down' || pat === 'nod') s += 2;
      if (f.extCount <= 2) s++;
      if (moving) s++;
      if (f.centroid.y > 0.3) s++;
      add('NEED', s, x);
    }

    // HELP — fist on palm moving up
    { let s = 0, x = 9;
      if (f.extCount <= 2) s++;
      if (traj === 'up') s += 2;
      if (moving && spd > 0.02) s += 2;
      if (f.thumbExt || f.thumbOnSide) s += 2;
      if (f.centroid.y > 0.3 && f.centroid.y < 0.6) s++;
      if (!f.indexMiddleSpread) s++;
      add('HELP', s, x);
    }

    // ═══ PLACES & THINGS ═══

    // HOME — closed hand touching cheek then jaw (fingertips to cheek area)
    { let s = 0, x = 8;
      if (f.thumbIndexTouch || f.thumbIndexDist < 0.04) s += 2;
      if (f.centroid.y < 0.45) s += 2; // face level
      if (f.extCount <= 2) s++;
      if (traj === 'down' || (moving && spd < 0.04)) s++;
      if (f.curls[1] > 70) s++;
      if (f.centroid.x > 0.3 && f.centroid.x < 0.7) s++;
      add('HOME', s, x);
    }

    // SCHOOL — clapping motion (one hand flat)
    { let s = 0, x = 8;
      if (f.extCount >= 4) s += 2;
      if (f.palmFacing) s++;
      if (pat === 'nod' || (moving && spd > 0.02 && spd < 0.08)) s += 2;
      if (f.centroid.y > 0.35 && f.centroid.y < 0.6) s++;
      if (f.indexExt && f.middleExt && f.ringExt) s++;
      if (!f.thumbAcross) s++;
      add('SCHOOL', s, x);
    }

    // FOOD — same as EAT but with different priority/context
    // (handled by EAT sign — they map to same gesture in ASL)

    // WATER — W hand tapping chin (3 fingers up at chin)
    { let s = 0, x = 9;
      if (f.indexExt && f.middleExt && f.ringExt) s += 2;
      if (!f.pinkyExt) s++;
      if (!f.thumbExt || f.thumbTucked) s++;
      if (f.centroid.y < 0.4) s += 2; // chin level
      if (pat === 'nod' || (moving && spd < 0.04)) s++;
      if (f.extCount === 3) s++;
      if (f.centroid.x > 0.3 && f.centroid.x < 0.7) s++;
      add('WATER', s, x);
    }

    // ═══ DESCRIPTORS ═══

    // GOOD — flat hand from chin forward
    { let s = 0, x = 9;
      if (f.extCount >= 3) s++;
      if (f.palmFacing) s++;
      if (f.centroid.y < 0.5) s += 2; // starts at chin
      if (traj === 'down' && spd > 0.01 && spd < 0.06) s += 2;
      if (!moving || spd < 0.06) s++;
      if (f.thumbExt) s++;
      if (!f.thumbAcross) s++;
      add('GOOD', s, x);
    }

    // BAD — flat hand from chin, flipping down
    { let s = 0, x = 8;
      if (f.extCount >= 3) s++;
      if (f.centroid.y < 0.5) s++;
      if (traj === 'down') s += 2;
      if (moving && spd > 0.02) s++;
      if (!f.palmFacing) s++; // palm faces down after flip
      if (f.thumbExt) s++;
      if (f.indexExt && f.middleExt) s++;
      add('BAD', s, x);
    }

    // HAPPY — open palm brushing up on chest
    { let s = 0, x = 8;
      if (f.extCount >= 4) s += 2;
      if (f.palmFacing) s++;
      if (traj === 'up') s += 2;
      if (f.centroid.y > 0.35 && f.centroid.y < 0.6) s++;
      if (moving && spd > 0.015 && spd < 0.07) s++;
      if (f.thumbExt) s++;
      add('HAPPY', s, x);
    }

    // SAD — open hands moving down in front of face
    { let s = 0, x = 8;
      if (f.extCount >= 3) s++;
      if (f.palmFacing) s++;
      if (traj === 'down') s += 2;
      if (f.centroid.y < 0.55) s++;
      if (moving && spd > 0.01 && spd < 0.06) s += 2;
      if (f.indexExt && f.middleExt) s++;
      add('SAD', s, x);
    }

    // ═══ RELATIONSHIPS ═══

    // LOVE — arms crossed over chest (fists crossed)
    { let s = 0, x = 8;
      if (f.extCount <= 1) s += 2;
      if (!f.indexExt && !f.middleExt && !f.ringExt && !f.pinkyExt) s += 2;
      if (f.centroid.y > 0.35 && f.centroid.y < 0.65) s++; // chest
      if (f.centroid.x > 0.3 && f.centroid.x < 0.7) s++;
      if (!moving || spd < 0.02) s++;
      if (f.thumbTucked || f.thumbOnSide) s++;
      add('LOVE', s, x);
    }

    // FRIEND — index fingers interlocking (hook shape)
    { let s = 0, x = 8;
      if (f.indexExt) s++;
      if (f.curls[1] > 40 && f.curls[1] < 130) s++; // hooked
      if (!f.middleExt && !f.ringExt && !f.pinkyExt) s++;
      if (f.centroid.y > 0.35 && f.centroid.y < 0.6) s++;
      if (!moving || spd < 0.03) s += 2;
      if (f.extCount <= 2) s++;
      if (f.thumbExt || f.thumbOnSide) s++;
      add('FRIEND', s, x);
    }

    // FAMILY — both hands circle (F handshape moving in circle)
    { let s = 0, x = 8;
      if (f.thumbIndexTouch || f.thumbIndexDist < 0.04) s += 2;
      if (f.middleExt && f.ringExt && f.pinkyExt) s++;
      if (moving && spd > 0.01 && spd < 0.07) s += 2;
      if (f.centroid.y > 0.3 && f.centroid.y < 0.6) s++;
      if (f.extCount >= 3) s++;
      if (pat !== 'wave') s++;
      add('FAMILY', s, x);
    }

    // ═══ TIME ═══

    // TODAY — both hands dropping (Y hands dropping)
    { let s = 0, x = 8;
      if (f.thumbExt && f.pinkyExt) s += 2;
      if (!f.indexExt && !f.middleExt && !f.ringExt) s++;
      if (traj === 'down') s += 2;
      if (f.centroid.y > 0.3) s++;
      if (moving && spd > 0.01) s++;
      if (f.extCount === 2) s++;
      add('TODAY', s, x);
    }

    // TOMORROW — thumb on chin moving forward (A hand at chin, arc forward)
    { let s = 0, x = 8;
      if (f.thumbExt) s += 2;
      if (!f.indexExt && !f.middleExt && !f.ringExt && !f.pinkyExt) s++;
      if (f.centroid.y < 0.45) s++; // face level
      if (traj === 'right' || pat === 'sweep-right' || pat === 'push') s += 2;
      if (moving && spd > 0.01) s++;
      if (f.extCount <= 2) s++;
      add('TOMORROW', s, x);
    }

    // MORE — both hands together, fingertips touching (flat O meeting)
    { let s = 0, x = 8;
      if (f.thumbIndexTouch || f.thumbIndexDist < 0.04) s += 2;
      if (f.curls[1] > 60) s++;
      if (f.curls[2] > 60) s++;
      if (f.centroid.y > 0.3 && f.centroid.y < 0.6) s++;
      if (!moving || spd < 0.03) s += 2;
      if (f.extCount <= 2) s++;
      add('MORE', s, x);
    }

    // Sort by confidence
    m.sort((a, b) => b.confidence - a.confidence);

    // Apply disambiguation for confusable pairs
    this._disambiguate(m, f, mo);

    // Filter: only return if top match is strong enough
    if (m.length > 0 && m[0].confidence < 55) {
      m[0].sign = 'UNCLEAR';
      m[0].confidence = 30;
    }

    return m.slice(0, 4);
  }

  _disambiguate(m, f, mo) {
    if (m.length < 2) return;
    const t = [m[0]?.sign, m[1]?.sign];

    // STOP vs HELLO — HELLO has wave, STOP is static push
    if (t.includes('STOP') && t.includes('HELLO')) {
      if ((mo?.pattern || '') === 'wave') this._boost(m, 'HELLO', 15);
      else this._boost(m, 'STOP', 12);
    }

    // YES vs SORRY — YES nods, SORRY circles
    if (t.includes('YES') && t.includes('SORRY')) {
      if ((mo?.pattern || '') === 'nod') this._boost(m, 'YES', 15);
      else this._boost(m, 'SORRY', 10);
    }

    // EAT vs HOME — both near face, EAT has more movement
    if (t.includes('EAT') && t.includes('HOME')) {
      if ((mo?.speed || 0) > 0.02) this._boost(m, 'EAT', 10);
      else this._boost(m, 'HOME', 10);
    }

    // GOOD vs THANK YOU — similar motion, THANK YOU more forward
    if (t.includes('GOOD') && t.includes('THANK YOU')) {
      if ((mo?.speed || 0) > 0.03) this._boost(m, 'THANK YOU', 10);
      else this._boost(m, 'GOOD', 10);
    }

    // I vs YOU — I points inward, YOU points outward
    if (t.includes('I') && t.includes('YOU')) {
      if (f.indexPointUp) this._boost(m, 'I', 12);
      else this._boost(m, 'YOU', 12);
    }

    // GO vs COME — GO sweeps away, COME pulls toward
    if (t.includes('GO') && t.includes('COME')) {
      if ((mo?.pattern || '') === 'pull') this._boost(m, 'COME', 15);
      else this._boost(m, 'GO', 12);
    }

    m.sort((a, b) => b.confidence - a.confidence);
  }

  _boost(m, sign, amt) {
    const x = m.find(i => i.sign === sign);
    if (x) x.confidence = Math.min(98, x.confidence + amt);
  }
}
