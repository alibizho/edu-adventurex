import type { ConceptId } from "../concepts/concepts.types";
import type {
  KnowledgeGapTemplate,
  StudyDocument,
  StudyModule,
  StudyToolConfig,
  TeachingScenario,
} from "./study.types";

const STUDY_TOOLS: readonly StudyToolConfig[] = [
  { id: "map", label: "Map" },
  { id: "tutorial", label: "Tutorial" },
  { id: "reset", label: "Reset" },
];

const DEFAULT_STUDENT = { name: "AI STUDENT", readiness: 0 } as const;

const DEFAULT_STATUS = {
  learningLabel: "LEARNING_MATERIAL_PHASE",
  readyLabel: "READY_TO_TEACH",
  meta: "CPU: 42%   MEM: 128KB",
} as const;

type StudyModuleInput = {
  conceptId: ConceptId;
  title: string;
  document: Omit<StudyDocument, "page" | "pageCount" | "readyLabel" | "completedLabel">;
  teaching: TeachingScenario;
};

type GapTopic = Omit<KnowledgeGapTemplate, "id" | "severity">;

function createStudyModule({ conceptId, title, document, teaching }: StudyModuleInput): StudyModule {
  return {
    conceptId,
    moduleLabel: "CURRENT MODULE",
    title,
    student: { ...DEFAULT_STUDENT },
    tools: STUDY_TOOLS,
    status: { ...DEFAULT_STATUS },
    document: {
      ...document,
      page: 1,
      pageCount: 1,
      readyLabel: "READY TO TEACH",
      completedLabel: "READY",
    },
    teaching,
  };
}

function createTeachingScenario(
  initialQuestion: string,
  focus: string,
  topics: readonly [GapTopic, GapTopic, GapTopic],
  explanation: string,
): TeachingScenario {
  const severities = ["HIGH", "MEDIUM", "LOW"] as const;
  return {
    initialQuestion,
    followUps: [
      `CAN YOU GIVE A CONCRETE EXAMPLE OF ${focus}?`,
      `WHAT IS THE MOST IMPORTANT CAUSE-AND-EFFECT RELATIONSHIP IN ${focus}?`,
      `HOW WOULD YOU EXPLAIN ${focus} WITHOUT USING TECHNICAL JARGON?`,
    ],
    fallbackQuestion: "CAN YOU EXPLAIN THAT AGAIN FROM A DIFFERENT ANGLE?",
    gaps: topics.map((topic, index) => ({
      ...topic,
      id: `${index + 1}-${topic.title.toLowerCase().replaceAll(" ", "-")}`,
      severity: severities[index],
    })),
    unknownHelp: {
      response: "I don't know...",
      explanation,
      acknowledgementLabel: "I GOT IT!",
    },
  };
}

export const STUDY_MODULES: Record<ConceptId, StudyModule> = {
  fundamentals: createStudyModule({
    conceptId: "fundamentals",
    title: "QUANTUM FUNDAMENTALS",
    document: {
      id: "QF_001_FOUNDATIONS",
      title: "WAVE-PARTICLE DUALITY",
      introduction: "Quantum objects can display both wave-like and particle-like behavior depending on how they are observed.",
      detail: "This dual behavior is a foundation of quantum theory and explains why classical intuition cannot fully describe microscopic systems.",
      figureCaption: "FIG 1.0: WAVE AND PARTICLE STATES",
    },
    teaching: createTeachingScenario(
      "WHY CAN THE SAME QUANTUM OBJECT BEHAVE LIKE BOTH A WAVE AND A PARTICLE?",
      "WAVE-PARTICLE DUALITY",
      [
        { title: "MEASUREMENT CONTEXT", whyItMatters: "THE EXPLANATION MAY NOT CONNECT THE OBSERVED BEHAVIOR TO THE EXPERIMENTAL SETUP.", recommendedReview: "REVIEW HOW DIFFERENT MEASUREMENTS REVEAL WAVE-LIKE OR PARTICLE-LIKE RESULTS." },
        { title: "INTERFERENCE", whyItMatters: "THE ROLE OF OVERLAPPING QUANTUM PATHS MAY REMAIN UNCLEAR.", recommendedReview: "REVIEW HOW PROBABILITY AMPLITUDES CREATE AN INTERFERENCE PATTERN." },
        { title: "CLASSICAL ANALOGIES", whyItMatters: "A CLASSICAL COMPARISON CAN HIDE WHAT IS UNIQUELY QUANTUM.", recommendedReview: "SEPARATE HELPFUL ANALOGIES FROM CLAIMS ABOUT THE ACTUAL QUANTUM STATE." },
      ],
      "A quantum object is described by a state that can produce wave-like interference while still being detected as a single, localized event. The experimental setup decides which aspect becomes visible. When paths remain indistinguishable, their probability amplitudes combine and create interference. When a detector records which path the object took, that physical interaction removes the coherent relationship between the paths, so the particle-like result remains. The object is not switching between two ordinary identities; wave and particle are complementary ways the same quantum state appears under different measurements.",
    ),
  }),
  ethics: createStudyModule({
    conceptId: "ethics",
    title: "ETHICS OF OBSERVATION",
    document: {
      id: "QE_001_RESPONSIBILITY",
      title: "THE RESPONSIBLE OBSERVER",
      introduction: "Quantum technologies create new responsibilities around privacy, security, and the interpretation of uncertain results.",
      detail: "Responsible practice requires transparent assumptions, careful communication, and clear limits on how measurements are used.",
      figureCaption: "FIG 1.0: OBSERVATION AND RESPONSIBILITY",
    },
    teaching: createTeachingScenario(
      "WHAT MAKES THE USE OF QUANTUM MEASUREMENT AN ETHICAL RESPONSIBILITY?",
      "RESPONSIBLE QUANTUM OBSERVATION",
      [
        { title: "PRIVACY IMPACT", whyItMatters: "THE EXPLANATION MAY OMIT WHO CAN BE HARMED BY SENSITIVE MEASUREMENT DATA.", recommendedReview: "REVIEW HOW QUANTUM SENSING AND COMPUTATION CAN AFFECT PRIVACY." },
        { title: "UNCERTAINTY", whyItMatters: "UNCERTAIN RESULTS SHOULD NOT BE COMMUNICATED AS CERTAIN FACTS.", recommendedReview: "REVIEW HOW TO STATE ASSUMPTIONS, ERROR RANGES, AND LIMITATIONS." },
        { title: "ACCOUNTABILITY", whyItMatters: "RESPONSIBILITY FOR HOW RESULTS ARE USED MAY BE LEFT UNDEFINED.", recommendedReview: "IDENTIFY WHO COLLECTS, INTERPRETS, AND ACTS ON THE MEASUREMENT." },
      ],
      "Quantum observation becomes an ethical responsibility when measurements or calculations can affect people, institutions, or sensitive information. A responsible observer explains uncertainty, protects private data, and avoids presenting a probabilistic result as a guaranteed fact. They also make clear who selected the assumptions, who can access the result, and who is accountable for decisions based on it. The physics does not supply the ethical rule by itself; people must decide how the technology is used, communicate its limits honestly, and consider who may benefit or be harmed.",
    ),
  }),
  "advanced-logic": createStudyModule({
    conceptId: "advanced-logic",
    title: "QUANTUM LOGIC",
    document: {
      id: "QL_001_SUPERPOSITION",
      title: "LOGIC BEYOND BINARY STATES",
      introduction: "Quantum logic allows a system to be described through combinations of states rather than a single classical true-or-false value.",
      detail: "Superposition and measurement change how propositions are combined, tested, and interpreted in a quantum system.",
      figureCaption: "FIG 1.0: SUPERPOSITION STATE MODEL",
    },
    teaching: createTeachingScenario(
      "WHY IS QUANTUM LOGIC NOT LIMITED TO A SINGLE TRUE-OR-FALSE STATE?",
      "QUANTUM LOGIC",
      [
        { title: "SUPERPOSITION", whyItMatters: "THE EXPLANATION MAY TREAT SUPERPOSITION AS ORDINARY UNCERTAINTY.", recommendedReview: "REVIEW HOW A QUANTUM STATE COMBINES AMPLITUDES BEFORE MEASUREMENT." },
        { title: "MEASUREMENT RULES", whyItMatters: "THE TRANSITION FROM A QUANTUM DESCRIPTION TO AN OBSERVED RESULT MAY BE MISSING.", recommendedReview: "REVIEW HOW MEASUREMENT MAPS A STATE TO POSSIBLE OUTCOMES." },
        { title: "BINARY COMPARISON", whyItMatters: "THE DIFFERENCE BETWEEN A QUBIT AND A CLASSICAL BIT MAY BE TOO VAGUE.", recommendedReview: "COMPARE CLASSICAL BOOLEAN STATES WITH A QUBIT STATE STEP BY STEP." },
      ],
      "Classical logic assigns a bit one definite value, zero or one. A qubit is described by amplitudes for both possible outcomes, including their relative phase. Before measurement, those amplitudes can interfere, so quantum operations must track more than a single true-or-false value. Measurement then produces one classical result according to the state's probabilities. Quantum logic is therefore not a claim that a statement is casually both true and false; it is a mathematical framework for reasoning about superposition, compatible measurements, and the outcomes that a particular experiment can reveal.",
    ),
  }),
  "practical-application": createStudyModule({
    conceptId: "practical-application",
    title: "QUANTUM APPLICATIONS",
    document: {
      id: "QA_001_COMPUTING",
      title: "FROM QUBITS TO COMPUTATION",
      introduction: "Quantum computers use qubits to represent and process combinations of states.",
      detail: "Useful algorithms depend on controlled superposition, interference, and measurement to amplify meaningful results.",
      figureCaption: "FIG 1.0: QUANTUM COMPUTATION FLOW",
    },
    teaching: createTeachingScenario(
      "HOW DOES A QUBIT HELP A QUANTUM COMPUTER PROCESS INFORMATION?",
      "QUANTUM COMPUTATION",
      [
        { title: "QUBIT STATE", whyItMatters: "THE EXPLANATION MAY DESCRIBE A QUBIT AS SIMPLY HOLDING TWO CLASSICAL VALUES.", recommendedReview: "REVIEW AMPLITUDES, PHASE, AND THE BLOCH-SPHERE DESCRIPTION." },
        { title: "INTERFERENCE", whyItMatters: "THE MECHANISM THAT AMPLIFIES USEFUL RESULTS MAY BE MISSING.", recommendedReview: "REVIEW HOW QUANTUM ALGORITHMS USE CONSTRUCTIVE AND DESTRUCTIVE INTERFERENCE." },
        { title: "MEASUREMENT OUTPUT", whyItMatters: "THE LIMITS OF READING A QUANTUM STATE MAY BE UNCLEAR.", recommendedReview: "REVIEW WHY MEASUREMENT RETURNS CLASSICAL OUTCOMES RATHER THAN THE FULL STATE." },
      ],
      "A qubit helps a quantum computer by storing amplitudes and phase for two basis states. Gates transform those quantities in a controlled way, and many qubits can form correlated states that cannot be represented as independent classical bits. A useful algorithm does not simply try every answer and read them all. It arranges interference so that amplitudes for useful outcomes grow while others cancel. Measurement finally returns an ordinary classical result, so the algorithm must shape the quantum state before it is read. The advantage comes from controlling superposition, entanglement, and interference together.",
    ),
  }),
  "case-studies": createStudyModule({
    conceptId: "case-studies",
    title: "QUANTUM CASE STUDIES",
    document: {
      id: "QC_001_DOUBLE_SLIT",
      title: "THE DOUBLE-SLIT EXPERIMENT",
      introduction: "The double-slit experiment reveals interference when quantum particles travel through multiple possible paths.",
      detail: "When path information is measured, the interference pattern changes, making the experiment a clear study of observation and state.",
      figureCaption: "FIG 1.0: DOUBLE-SLIT OBSERVATION",
    },
    teaching: createTeachingScenario(
      "WHAT DOES THE DOUBLE-SLIT EXPERIMENT SHOW ABOUT QUANTUM BEHAVIOR?",
      "THE DOUBLE-SLIT EXPERIMENT",
      [
        { title: "PATH INFORMATION", whyItMatters: "THE EXPLANATION MAY NOT SHOW WHY KNOWING THE PATH CHANGES THE RESULT.", recommendedReview: "REVIEW THE DIFFERENCE BETWEEN MEASURED AND UNMEASURED PATH INFORMATION." },
        { title: "INTERFERENCE PATTERN", whyItMatters: "THE CONNECTION BETWEEN MANY DETECTIONS AND THE FINAL PATTERN MAY BE MISSING.", recommendedReview: "REVIEW HOW INDIVIDUAL EVENTS BUILD A STATISTICAL INTERFERENCE PATTERN." },
        { title: "DECOHERENCE", whyItMatters: "ENVIRONMENTAL INTERACTION MAY BE CONFUSED WITH HUMAN OBSERVATION.", recommendedReview: "REVIEW HOW PHYSICAL INTERACTIONS DESTROY COHERENT PATH RELATIONSHIPS." },
      ],
      "In the double-slit experiment, individual quantum particles arrive at the screen as localized detections, but many detections build an interference pattern when no usable path information exists. The pattern shows that the alternatives through the two slits combine as probability amplitudes. If a detector physically records which slit each particle passes through, that interaction makes the paths distinguishable and destroys the interference. A conscious person is not required. The important change comes from the measurement interaction and the loss of coherence between the possible paths.",
    ),
  }),
  subject: createStudyModule({
    conceptId: "subject",
    title: "QUANTUM PHYSICS",
    document: {
      id: "QP_001_INTRO",
      title: "THE OBSERVER EFFECT",
      introduction: "In quantum mechanics, the observer effect is the theory that the mere observation of a phenomenon inevitably changes that phenomenon.",
      detail: "Schrödinger's Cat is a thought experiment that illustrates a paradox of quantum superposition. In this state, a system exists in all possible states simultaneously until it is measured.",
      figureCaption: "FIG 1.0: CONCEPTUAL DIAGRAM OF MEASUREMENT",
    },
    teaching: {
      initialQuestion: "If I just look at the particles, they change?\nWhy does that happen?",
      followUps: [
        "WHAT IS THE DIFFERENCE BETWEEN A HUMAN OBSERVER AND A PHYSICAL DETECTOR?",
        "HOW DOES SUPERPOSITION RELATE TO THE POSSIBLE MEASUREMENT OUTCOMES?",
        "CAN YOU EXPLAIN MEASUREMENT COLLAPSE WITHOUT SAYING THAT CONSCIOUSNESS CAUSES IT?",
      ],
      fallbackQuestion: "CAN YOU EXPLAIN THAT AGAIN FROM A DIFFERENT ANGLE?",
      unknownHelp: {
        response: "I don't know...",
        explanation: "Okay...According to the observer effect in quantum physics, simply looking at particles does not mean that human awareness changes them. Instead, the act of measuring a particle requires interacting with it, such as using light or another particle to detect it. This interaction can disturb the particle and change its state. In quantum mechanics, particles can exist in a range of possible states until they are measured, and the measurement forces the system to show one specific outcome. So, it is not ‘looking’ that changes particles, but the physical interaction involved in observing them.",
        acknowledgementLabel: "I GOT IT!",
      },
      gaps: [
        { id: "measurement-collapse", title: "MEASUREMENT COLLAPSE", severity: "HIGH", whyItMatters: "THE EXPLANATION CONFUSES OBSERVATION WITH PHYSICAL MEASUREMENT.", recommendedReview: "REVIEW HOW DETECTORS INTERACT WITH A QUANTUM SYSTEM BEFORE EXPLAINING COLLAPSE." },
        { id: "superposition", title: "SUPERPOSITION", severity: "MEDIUM", whyItMatters: "THE RELATIONSHIP BETWEEN POSSIBLE STATES AND OBSERVED OUTCOMES MAY BE INCOMPLETE.", recommendedReview: "REVIEW HOW SUPERPOSITION IS REPRESENTED BEFORE AND AFTER MEASUREMENT." },
        { id: "observer-vs-detector", title: "OBSERVER VS DETECTOR", severity: "LOW", whyItMatters: "THE WORD OBSERVER MAY SOUND LIKE IT REQUIRES HUMAN CONSCIOUSNESS.", recommendedReview: "DISTINGUISH A PHYSICAL MEASUREMENT INTERACTION FROM A PERSON READING THE RESULT." },
      ],
    },
  }),
  tools: createStudyModule({
    conceptId: "tools",
    title: "QUANTUM TOOLS",
    document: {
      id: "QT_001_INSTRUMENTS",
      title: "MEASURING THE INVISIBLE",
      introduction: "Quantum experiments rely on detectors, lasers, cryogenic systems, and precise control electronics.",
      detail: "Each instrument converts fragile quantum behavior into signals that can be recorded, compared, and interpreted.",
      figureCaption: "FIG 1.0: QUANTUM MEASUREMENT TOOLCHAIN",
    },
    teaching: createTeachingScenario(
      "HOW DO QUANTUM INSTRUMENTS TURN AN INVISIBLE STATE INTO A MEASURABLE SIGNAL?",
      "QUANTUM MEASUREMENT TOOLS",
      [
        { title: "DETECTOR INTERACTION", whyItMatters: "THE PHYSICAL INTERACTION THAT PRODUCES A SIGNAL MAY BE MISSING.", recommendedReview: "REVIEW HOW A DETECTOR COUPLES TO A QUANTUM SYSTEM." },
        { title: "NOISE CONTROL", whyItMatters: "THE EXPLANATION MAY ASSUME EVERY RECORDED SIGNAL IS THE QUANTUM EFFECT ITSELF.", recommendedReview: "REVIEW CALIBRATION, BACKGROUND NOISE, AND REPEATED MEASUREMENTS." },
        { title: "SIGNAL INTERPRETATION", whyItMatters: "THE STEP FROM RAW SIGNAL TO SCIENTIFIC CLAIM MAY BE UNCLEAR.", recommendedReview: "TRACE HOW EXPERIMENTAL DATA IS PROCESSED AND COMPARED WITH A MODEL." },
      ],
      "Quantum instruments turn an invisible state into a measurable signal by coupling the system to a detector. A laser, sensor, or readout circuit interacts with the quantum object and converts a tiny physical change into an electrical or optical record. Because this process can disturb the state, experiments control temperature, vibration, electromagnetic noise, and timing very carefully. Researchers repeat and calibrate the measurement, compare it with background noise, and use a model to interpret the data. The recorded signal is evidence produced by a physical interaction, not a direct picture of the untouched state.",
    ),
  }),
  history: createStudyModule({
    conceptId: "history",
    title: "HISTORY OF QUANTUM THEORY",
    document: {
      id: "QH_001_ORIGINS",
      title: "FROM PLANCK TO BELL",
      introduction: "Quantum theory grew from attempts to explain radiation, atomic spectra, and the behavior of light.",
      detail: "The work of Planck, Einstein, Bohr, Schrödinger, and Bell transformed those puzzles into a new framework for physical reality.",
      figureCaption: "FIG 1.0: QUANTUM THEORY TIMELINE",
    },
    teaching: createTeachingScenario(
      "HOW DID EARLY QUANTUM DISCOVERIES CHANGE THE CLASSICAL VIEW OF PHYSICS?",
      "THE HISTORY OF QUANTUM THEORY",
      [
        { title: "PLANCK'S QUANTA", whyItMatters: "THE STARTING PROBLEM OF BLACK-BODY RADIATION MAY BE OMITTED.", recommendedReview: "REVIEW WHY PLANCK INTRODUCED DISCRETE ENERGY PACKETS." },
        { title: "MODEL EVOLUTION", whyItMatters: "THE THEORY MAY SOUND LIKE ONE SINGLE DISCOVERY RATHER THAN A SERIES OF REVISIONS.", recommendedReview: "REVIEW THE SEQUENCE FROM PLANCK AND EINSTEIN TO BOHR AND SCHRÖDINGER." },
        { title: "BELL'S RESULT", whyItMatters: "THE LATER TESTS OF QUANTUM CORRELATIONS MAY BE DISCONNECTED FROM THE EARLY HISTORY.", recommendedReview: "REVIEW HOW BELL TURNED FOUNDATIONAL QUESTIONS INTO TESTABLE INEQUALITIES." },
      ],
      "Quantum theory developed through a sequence of problems that classical physics could not solve. Planck introduced discrete energy packets to explain thermal radiation. Einstein used light quanta to explain the photoelectric effect, while Bohr applied quantized states to atoms. Schrödinger, Heisenberg, and Born then built a broader mathematical theory of states, observables, and probabilities. Later, Bell transformed arguments about hidden variables and quantum correlations into inequalities that experiments could test. The modern theory was therefore not one sudden idea, but a framework refined by predictions, disagreements, and increasingly precise experiments.",
    ),
  }),
};

export const DEFAULT_STUDY_CONCEPT_ID: ConceptId = "subject";

export function isConceptId(value: string | null | undefined): value is ConceptId {
  return Boolean(value && Object.prototype.hasOwnProperty.call(STUDY_MODULES, value));
}

export function resolveStudyConceptId(
  queryConceptId: string | null,
  routeConceptId?: string,
): ConceptId {
  if (isConceptId(queryConceptId)) return queryConceptId;
  if (isConceptId(routeConceptId)) return routeConceptId;
  return DEFAULT_STUDY_CONCEPT_ID;
}

export function getStudyModule(conceptId: ConceptId): StudyModule {
  return STUDY_MODULES[conceptId];
}
