// SentenceBuilder v2 — Optimized for controlled vocabulary
// Converts ASL gloss (word list) into grammatically correct English
// Tuned for the 30-word vocabulary set

const GRAMMAR = {
  futureMarkers: ['TOMORROW', 'LATER', 'SOON', 'WILL'],
  pastMarkers: ['YESTERDAY', 'BEFORE', 'ALREADY'],
  presentMarkers: ['NOW', 'TODAY'],

  // Bigram → expansion (preposition/article insertion)
  expansions: {
    'GO HOME': 'GO HOME',
    'GO SCHOOL': 'GO TO SCHOOL',
    'COME HOME': 'COME HOME',
    'WANT EAT': 'WANT TO EAT',
    'WANT DRINK': 'WANT TO DRINK',
    'WANT GO': 'WANT TO GO',
    'WANT COME': 'WANT TO COME',
    'WANT HELP': 'WANT HELP',
    'NEED EAT': 'NEED TO EAT',
    'NEED DRINK': 'NEED TO DRINK',
    'NEED GO': 'NEED TO GO',
    'NEED HELP': 'NEED HELP',
    'NEED WATER': 'NEED WATER',
    'NEED FOOD': 'NEED FOOD',
    'LIKE EAT': 'LIKE TO EAT',
    'LIKE DRINK': 'LIKE TO DRINK',
    'LOVE FOOD': 'LOVE FOOD',
    'LOVE FAMILY': 'LOVE MY FAMILY',
    'LOVE FRIEND': 'LOVE MY FRIEND',
    'I HAPPY': 'I AM HAPPY',
    'I SAD': 'I AM SAD',
    'I GOOD': 'I AM GOOD',
    'I BAD': 'I AM BAD',
    'I SORRY': 'I AM SORRY',
    'I HUNGRY': 'I AM HUNGRY',
    'I TIRED': 'I AM TIRED',
    'YOU GOOD': 'YOU ARE GOOD',
    'YOU HAPPY': 'YOU ARE HAPPY',
    'YOU SAD': 'YOU ARE SAD',
    'I WANT': 'I WANT',
    'I NEED': 'I NEED',
    'I GO': 'I GO',
    'I COME': 'I COME',
    'YOU WANT': 'YOU WANT',
    'YOU NEED': 'YOU NEED',
    'THANK YOU': 'THANK YOU',
    'MORE FOOD': 'MORE FOOD',
    'MORE WATER': 'MORE WATER',
  },

  // Verb tense conjugation
  verbs: {
    'GO':    { present: 'go', past: 'went', future: 'will go' },
    'COME':  { present: 'come', past: 'came', future: 'will come' },
    'EAT':   { present: 'eat', past: 'ate', future: 'will eat' },
    'DRINK': { present: 'drink', past: 'drank', future: 'will drink' },
    'WANT':  { present: 'want', past: 'wanted', future: 'will want' },
    'NEED':  { present: 'need', past: 'needed', future: 'will need' },
    'HELP':  { present: 'help', past: 'helped', future: 'will help' },
    'STOP':  { present: 'stop', past: 'stopped', future: 'will stop' },
    'LOVE':  { present: 'love', past: 'loved', future: 'will love' },
    'LIKE':  { present: 'like', past: 'liked', future: 'will like' },
  },

  // Word → English mapping
  wordMap: {
    'I': 'I', 'ME': 'I', 'YOU': 'you',
    'HELLO': 'hello', 'PLEASE': 'please', 'THANK YOU': 'thank you',
    'YES': 'yes', 'NO': 'no',
    'HOME': 'home', 'SCHOOL': 'school',
    'FOOD': 'food', 'WATER': 'water',
    'GOOD': 'good', 'BAD': 'bad',
    'HAPPY': 'happy', 'SAD': 'sad',
    'SORRY': 'sorry', 'MORE': 'more',
    'FRIEND': 'friend', 'FAMILY': 'family',
    'TODAY': 'today', 'TOMORROW': 'tomorrow',
    'AM': 'am', 'ARE': 'are', 'IS': 'is',
    'TO': 'to', 'MY': 'my', 'THE': 'the',
  },

  // Nouns needing articles when standalone
  nounsNeedArticle: ['SCHOOL', 'FOOD', 'WATER'],
};

export class SentenceBuilder {
  constructor() {
    this.glossWords = [];
    this.wordConfidences = [];
    this.history = [];
    this.maxHistory = 20;
  }

  addWord(word, confidence) {
    const w = word.toUpperCase();
    // Prevent exact consecutive duplicates
    if (this.glossWords.length > 0 && this.glossWords[this.glossWords.length - 1] === w) {
      return;
    }
    this.glossWords.push(w);
    this.wordConfidences.push(confidence || 70);
  }

  getGloss() {
    return this.glossWords.join(' ');
  }

  buildSentence() {
    if (this.glossWords.length === 0) return '';

    let words = [...this.glossWords];

    // 1. Detect tense
    let tense = 'present';
    for (const w of words) {
      if (GRAMMAR.futureMarkers.includes(w)) { tense = 'future'; break; }
      if (GRAMMAR.pastMarkers.includes(w)) { tense = 'past'; break; }
    }

    // 2. Remove implicit tense markers
    words = words.filter(w => w !== 'WILL');

    // 3. Expand bigrams (preposition/copula insertion)
    words = this._expandBigrams(words);

    // 4. Convert to English
    const result = words.map(w => {
      // Verb conjugation
      const verb = GRAMMAR.verbs[w];
      if (verb) return verb[tense] || verb.present;

      // Direct mapping
      if (GRAMMAR.wordMap[w]) return GRAMMAR.wordMap[w];

      return w.toLowerCase();
    });

    // 5. Fix subject pronoun
    if (result[0] === 'me') result[0] = 'I';

    // 6. Capitalize + punctuation
    let sentence = result.join(' ');
    sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1);

    if (!/[.!?]$/.test(sentence)) {
      const qWords = ['what', 'where', 'when', 'who', 'why', 'how'];
      sentence += qWords.some(q => sentence.toLowerCase().startsWith(q)) ? '?' : '.';
    }

    // 7. Clean up spacing
    sentence = sentence.replace(/\s+/g, ' ').trim();

    return sentence;
  }

  _expandBigrams(words) {
    const result = [];
    let i = 0;
    while (i < words.length) {
      if (i < words.length - 1) {
        const bigram = words[i] + ' ' + words[i + 1];
        if (GRAMMAR.expansions[bigram]) {
          const expanded = GRAMMAR.expansions[bigram].split(' ');
          result.push(...expanded);
          i += 2;
          continue;
        }
      }
      result.push(words[i]);
      i++;
    }
    return result;
  }

  completeSentence() {
    if (this.glossWords.length === 0) return null;

    const sentence = this.buildSentence();
    const gloss = this.getGloss();
    const avgConf = this.wordConfidences.length > 0
      ? Math.round(this.wordConfidences.reduce((a, b) => a + b, 0) / this.wordConfidences.length)
      : 0;

    this.history.push(sentence);
    if (this.history.length > this.maxHistory) this.history.shift();

    this.glossWords = [];
    this.wordConfidences = [];

    return { sentence, gloss, confidence: avgConf };
  }

  getSentenceConfidence() {
    if (this.wordConfidences.length === 0) return 0;
    const avg = this.wordConfidences.reduce((a, b) => a + b, 0) / this.wordConfidences.length;
    const lengthBonus = Math.min(12, this.glossWords.length * 3);
    return Math.min(98, Math.round(avg + lengthBonus));
  }

  reset() {
    this.glossWords = [];
    this.wordConfidences = [];
    this.history = [];
  }
}
