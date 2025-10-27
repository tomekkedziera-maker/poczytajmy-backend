// verbs.js

export const VERBS_MOTION = [
  "iść","pójść","chodzić","jechać","pojechać","wracać","wrócić",
  "wejść","wyjść","wsiąść","wysiąść","dojść","podejść","przejść",
  "zajechać","dotrzeć","podjechać","odjechać","przyjść","biec","lecieć"
];

export const VERBS_PLACEBOUND = [
  "usiąść","siedzieć","siąść","stać","stanąć","leżeć","położyć się",
  "czekać","czytać","pisać","bawić się","grać","jeść","pić",
  "oglądać","rozmawiać","uczyć się","pracować","odpoczywać","spać","rysować"
];

export const VERBS_PERCEPTION = [
  "patrzeć","popatrzeć","spoglądać","spojrzeć","przyglądać się",
  "oglądać","zaglądać","zerkać","widzieć","słyszeć","czuć"
];

// (opcjonalnie) duża wspólna lista – może zawierać >200 słów
export const COMMON_VERBS = [
  ...VERBS_MOTION,
  ...VERBS_PLACEBOUND,
  ...VERBS_PERCEPTION,
];

// (opcjonalnie) default
export default COMMON_VERBS;
