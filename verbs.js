// verbs.js
// Zbiór ~220 najczęstszych czasowników w języku polskim — optymalny zakres dla tekstów dzieci do 13 lat.
// Używany w module comprehend (np. wykrywanie czynności w zdaniu)

export const COMMON_VERBS = [
  // Ruch i lokalizacja
  "iść", "pójść", "chodzić", "jechać", "pojechać", "biec", "biegć", "lecieć", "przylecieć", "wracać", "wrócić", "wchodzić", "wejść", "wychodzić", "wyjść", "wsiadać", "wsiąść", "wysiadać", "zjechać", "dojechać", "płynąć", "przepłynąć", "podróżować", "wędrować", "zatrzymać", "ruszyć", "stanąć", "stać", "usiąść", "siedzieć", "leżeć", "wstać", "wstawać",

  // Czynności codzienne
  "jeść", "zjeść", "pić", "wypić", "gotować", "piec", "sprzątać", "myć", "ubierać", "rozbierać", "czesać", "malować", "prasować", "kroić", "nosić", "pakować", "otwierać", "zamykać", "wkładać", "wyjmować", "kupować", "sprzedawać", "nosić", "podnosić", "trzymać", "brać", "dawać", "kłaść", "położyć", "rzucać", "rzucić",

  // Mówienie i słyszenie
  "mówić", "powiedzieć", "rozmawiać", "pytać", "odpowiadać", "krzyczeć", "szeptać", "śpiewać", "czytać", "czytać", "pisać", "słuchać", "wołać", "tłumaczyć", "rozumieć", "opowiadać", "głośno mówić",

  // Nauka i szkoła
  "uczyć się", "nauczyć się", "powtarzać", "liczyć", "rysować", "malować", "czytać", "pisać", "rozwiązywać", "notować", "czytać", "uczyć", "przypominać", "zapamiętać", "odrabiać", "pisać sprawdzian", "czytać książkę",

  // Emocje i stany
  "cieszyć się", "smucić się", "złościć się", "bać się", "śmiać się", "płakać", "kochać", "lubić", "nienawidzić", "cieszyć", "nudzić się", "zaskakiwać", "tęsknić", "martwić się", "cieszyć się", "zachwycać się",

  // Postrzeganie i zmysły
  "widzieć", "zobaczyć", "oglądać", "patrzeć", "spojrzeć", "słyszeć", "usłyszeć", "czuć", "dotykać", "powąchać", "smakować",

  // Praca i tworzenie
  "pracować", "budować", "naprawiać", "rysować", "tworzyć", "projektować", "zrobić", "robić", "składać", "układać", "tworzyć", "szyć", "konstruować",

  // Odpoczynek i zabawa
  "bawić się", "grać", "czytać", "oglądać", "spacerować", "tańczyć", "śpiewać", "rysować", "malować", "budować", "leżeć", "odpoczywać", "śmiać się", "rozmawiać", "grać w piłkę", "grać w gry", "oglądać bajki",

  // Przyroda i środowisko
  "rosnąć", "kwitnąć", "spadać", "świecić", "padać", "wiać", "topnieć", "zamarzać", "grzać", "mrugać", "świecić słońce", "śpiewać ptaki", "pływać ryby",

  // Myślenie i planowanie
  "myśleć", "zastanawiać się", "planować", "decydować", "marzyć", "wiedzieć", "znać", "rozumieć", "przypominać sobie", "zgadywać", "domyślać się",

  // Społeczne i rodzinne
  "pomagać", "prosić", "dziękować", "spotykać", "odwiedzać", "rozmawiać", "bawić się", "pracować", "świętować", "dzielić się", "przytulać", "pocieszać", "witać", "żegnać", "rozmawiać",

  // Inne przydatne
  "spaść", "podnieść", "upadać", "zginąć", "znaleźć", "szukać", "pojawiać się", "zniknąć", "stać się", "pozostać", "zmieniać", "zrobić się", "stać się",

  // Technologia i współczesność
  "uruchamiać", "włączać", "wyłączać", "nagrywać", "grać", "drukować", "pisać na komputerze", "czytać online", "wysyłać", "odbierać", "dzwonić", "nagrywać film", "robić zdjęcie"
];

export default COMMON_VERBS;
