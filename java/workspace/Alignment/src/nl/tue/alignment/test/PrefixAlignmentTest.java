package nl.tue.alignment.test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutionException;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.classification.XEventNameClassifier;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.factory.XFactory;
import org.deckfour.xes.factory.XFactoryRegistry;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.model.XAttributable;
import org.deckfour.xes.model.XAttribute;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.processmining.log.utils.XUtils;
import org.processmining.models.connections.GraphLayoutConnection;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetGraph;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.graphbased.directed.petrinet.impl.PetrinetFactory;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.pnml.base.FullPnmlElementFactory;
import org.processmining.plugins.pnml.base.Pnml;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserFactory;

import au.com.bytecode.opencsv.CSVReader;
import lpsolve.LpSolve;
import nl.tue.alignment.Progress;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.ReplayerParameters;
import nl.tue.alignment.Utils;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;

public class PrefixAlignmentTest {
//	private static final int THREADS = Runtime.getRuntime().availableProcessors() - 1;
	private static final int THREADS = 1;
	private static String SEP = Utils.SEP;
	private static String FOLDER = "/home/developer/data/BPM2018/correlation-tests/";
	private static String MODEL_FOLDER = FOLDER + "models/";
	private static String LOG_FOLDER = FOLDER + "logs/";
	private static String RESULT_FOLDER = "/home/developer/results/";

	static {
		try {
			System.loadLibrary("lpsolve55");
			System.loadLibrary("lpsolve55j");
		} catch (Exception e) {
			e.printStackTrace();
		}
		LpSolve.lpSolveVersion();
	}
	
	public static void main(String[] args) throws Exception {
		int timeoutPerTraceInSec = Integer.MAX_VALUE;
		mainFolder(Debug.STATS, timeoutPerTraceInSec, "8", "12", "14", "15", "19", 
				"26", "27", "32", "39", "43", "49", "50");
	}
	
	public static void mainFolder(Debug debug, int timeoutPerTraceInSec, String... eval) throws Exception {
		for (String netId : eval) {
			String[] pnmlFilenames = new File(MODEL_FOLDER).list(new FilenameFilter() {
				
				public boolean accept(File dir, String name) {
					return name.endsWith("_id_" + netId + ".pnml");
				}
				
			});
			
			if (pnmlFilenames.length != 1) {
				throw new IllegalArgumentException("More than one net with id: " + netId);
			}
			
			String pnmlFilename = pnmlFilenames[0];
					
			String[] logFilenames = new File(LOG_FOLDER).list(new FilenameFilter() {
				
				public boolean accept(File dir, String name) {
					return name.startsWith("log_" + pnmlFilename);
//					return name.startsWith("log_" + pnmlFilename + "_noise_trace_0.5_noise_event_0.5.csv");
				}
				
			});
			
			if (logFilenames.length < 1) {
				throw new IllegalArgumentException("No log for net: " + pnmlFilename);
			}
			
			for (String logFilename: logFilenames) {
				File pnmlFile = new File(MODEL_FOLDER + pnmlFilename);
				FileInputStream inputStream = new FileInputStream(pnmlFile);
				Pnml pnml = importPnmlFromStream(inputStream, pnmlFile.getName(), pnmlFile.length());
				inputStream.close();
				
				Petrinet net = PetrinetFactory.newPetrinet(pnml.getLabel());
				Marking initMarking = new Marking();
				GraphLayoutConnection layout = new GraphLayoutConnection(net);
				
				pnml.convertToNet(net, initMarking, layout);
				
				Marking finalMarking = getFinalMarking(net);
				
//				System.out.println("Number of places: " + net.getPlaces().size());
				
				// import log csv file
				FileReader fileReader = new FileReader(LOG_FOLDER + logFilename);
				char separator = '\t';
				CSVReader csvReader = new CSVReader(fileReader, separator);
				List<String[]> allLines = csvReader.readAll();
				allLines.remove(0); // skip header line
				
				// convert to prefix log
				XLog log = convertCSVToPrefixLog(allLines);
				
				// create transition event mapping
				XEventClassifier eventClassifier = new XEventNameClassifier();
				String filepath = RESULT_FOLDER + logFilename.replace(".csv", "");
				try {
					System.out.println("Replay experiment on " + logFilename);
					doReplayExperiment(debug, filepath, net, initMarking, finalMarking, log, eventClassifier, 
							timeoutPerTraceInSec);
				} catch (Exception e) {
					System.err.println("Exception: " + e.getMessage());
					e.printStackTrace();
				}
			}
		}
		
		System.exit(0);
	}
	
	private static void doReplayExperiment(Debug debug, String filepath, Petrinet net, Marking initMarking,
			Marking finalMarking, XLog log, XEventClassifier eventClassifier, int timeoutPerTraceInSec)
			throws FileNotFoundException, InterruptedException, ExecutionException {
		XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
		TransEvClassMapping mapping = constructMapping(net, log, dummyEvClass, eventClassifier);
		XLogInfo summary = XLogInfoFactory.createLogInfo(log, eventClassifier);
		XEventClasses classes = summary.getEventClasses();
		
		int threads = THREADS;
		
		int timeout = log.size() * timeoutPerTraceInSec * 1000 / 10;
		int maxNumberOfStates = Integer.MAX_VALUE;
		int costUpperBound = Integer.MAX_VALUE;
		boolean moveSort = false;
		boolean useInt = false;
		boolean partialOrder = false;
		boolean preferExact = true;
		boolean queueSort = true;
		boolean isPrefix = true;
		int maxReducedSequenceLength = 1;
		boolean buildFullStateSpace = false;
		
		ReplayerParameters parameters = new ReplayerParameters.Dijkstra(moveSort, queueSort, threads, debug, 
				timeout, maxNumberOfStates, costUpperBound, partialOrder, maxReducedSequenceLength, 
				buildFullStateSpace, isPrefix);
		doReplay(debug, filepath, net, initMarking, finalMarking, log, mapping, classes, parameters);
	}
	
	private static void doReplay(Debug debug, String filepath, PetrinetGraph net, Marking initMarking,
			Marking finalMarking, XLog log, TransEvClassMapping mapping, XEventClasses classes, 
			ReplayerParameters parameters) throws FileNotFoundException, InterruptedException, ExecutionException {
		PrintStream stream;
		if (debug == Debug.STATS || debug == Debug.NORMAL) {
			stream = new PrintStream(new File(filepath + ".csv"));
		} else if (debug == Debug.DOT) {
			stream = new PrintStream(new File(filepath + ".dot"));
		} else {
			stream = System.out;
		}
		ReplayAlgorithm.Debug.setOutputStream(stream);
		
		long start = System.nanoTime();
		boolean mergeDuplicateTraces = false;
		Replayer replayer = new Replayer(parameters, (Petrinet) net, initMarking, finalMarking, classes, mapping, mergeDuplicateTraces);
		PNRepResult result = replayer.computePNRepResult(Progress.INVISIBLE, log);
		long end = System.nanoTime();
		
		if (stream != System.out) {
			stream.close();
		}
		
		System.out.println("Took: " + String.format("%.3fms", (end - start) / 1000000.0));
	}
	
	private static XLog convertCSVToPrefixLog(List<String[]> lines) {
//		System.out.print("Converting CSV to prefix log...");
		XFactory factory = XFactoryRegistry.instance().currentDefault();
		XLog log = factory.createLog();
		String curCaseId = null;
		int curLength = 0;
		XTrace trace = factory.createTrace();
//		int i = 0;
		
		for (String[] row : lines) {
			String caseId = row[0];
			String eventClass = row[1];
			String eventId = row[2];
			
//			System.out.println("caseid: " + row[0] + ", " + "eventClass: " + row[1]);
//			System.out.println("curCaseId: " + curCaseId + ", caseId: " + caseId);
			
			if (curCaseId == null || !curCaseId.equals(caseId)) {
				trace = factory.createTrace();
				curCaseId = caseId;
				curLength = 0;
			} else {
//				System.out.println("Cloning trace");
				trace = (XTrace) trace.clone();
			}
			
			++curLength;
			assignName(factory, trace, curCaseId + "_" + curLength);
			
			XEvent event = factory.createEvent();
			assignName(factory, event, eventClass);
			
			trace.add(event);
			log.add(trace);
			
//			if (i++ >= 1) {
//				break;
//			}
		}
//		System.out.println("Done");
		return log;
	}
	
	private static void assignAttribute(XAttributable a, XAttribute value) {
		XUtils.putAttribute(a, value);
	}
	
	private static void assignName(XFactory factory, XAttributable a, String value) {
		assignAttribute(a, factory.createAttributeLiteral(XConceptExtension.KEY_NAME, value, XConceptExtension.instance()));
	}
	
	public static Pnml importPnmlFromStream(InputStream input, String filename, long fileSizeInBytes) throws Exception {
		FullPnmlElementFactory pnmlFactory = new FullPnmlElementFactory();
		/*
		 * Get an XML pull parser.
		 */
		XmlPullParserFactory factory = XmlPullParserFactory.newInstance();
		factory.setNamespaceAware(true);
		XmlPullParser xpp = factory.newPullParser();
		/*
		 * Initialize the parser on the provided input.
		 */
		xpp.setInput(input, null);
		/*
		 * Get the first event type.
		 */
		int eventType = xpp.getEventType();
		/*
		 * Create a fresh PNML object.
		 */
		Pnml pnml = new Pnml();
		synchronized (pnmlFactory) {
			pnml.setFactory(pnmlFactory);

			/*
			 * Skip whatever we find until we've found a start tag.
			 */
			while (eventType != XmlPullParser.START_TAG) {
				eventType = xpp.next();
			}
			/*
			 * Check whether start tag corresponds to PNML start tag.
			 */
			if (xpp.getName().equals(Pnml.TAG)) {
				/*
				 * Yes it does. Import the PNML element.
				 */
				pnml.importElement(xpp, pnml);
			} else {
				/*
				 * No it does not. Return null to signal failure.
				 */
				pnml.log(Pnml.TAG, xpp.getLineNumber(), "Expected pnml");
			}
		}

		/*
		 * Initialize necessary objects.
		 */
		Petrinet net = PetrinetFactory.newPetrinet(pnml.getLabel());
		Marking marking = new Marking();
		Collection<Marking> finalMarkings = new HashSet<Marking>();
		GraphLayoutConnection layout = new GraphLayoutConnection(net);
		/*
		 * Copy the imported data into these objects.
		 */
		return pnml;
	}

	private static Marking getFinalMarking(PetrinetGraph net) {
		Marking finalMarking = new Marking();

		for (Place p : net.getPlaces()) {
			if (net.getOutEdges(p).isEmpty())
				finalMarking.add(p);
		}

		return finalMarking;
	}

	private static Marking getInitialMarking(PetrinetGraph net) {
		Marking initMarking = new Marking();

		for (Place p : net.getPlaces()) {
			if (net.getInEdges(p).isEmpty())
				initMarking.add(p);
		}

		return initMarking;
	}

	private static TransEvClassMapping constructMapping(PetrinetGraph net, XLog log, XEventClass dummyEvClass,
			XEventClassifier eventClassifier) {
		TransEvClassMapping mapping = new TransEvClassMapping(eventClassifier, dummyEvClass);

		XLogInfo summary = XLogInfoFactory.createLogInfo(log, eventClassifier);

		for (Transition t : net.getTransitions()) {
			boolean mapped = false;
			for (XEventClass evClass : summary.getEventClasses().getClasses()) {
				String id = evClass.getId();

				if (t.getLabel().equals(id)) {
					mapping.put(t, evClass);
					mapped = true;
					break;
				} else if (id.equals(t.getLabel() + "+complete")) {
					mapping.put(t, evClass);
					mapped = true;
					break;
				} else if (id.equals(t.getLabel() + "+")) {
					mapping.put(t, evClass);
					mapped = true;
					break;
				}
			}

			if (!mapped && !t.isInvisible()) {
				mapping.put(t, dummyEvClass);
			}

		}

		return mapping;
	}
}
