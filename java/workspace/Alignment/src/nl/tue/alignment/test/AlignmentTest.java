package nl.tue.alignment.test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.concurrent.ExecutionException;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.classification.XEventNameClassifier;
import org.deckfour.xes.in.XMxmlParser;
import org.deckfour.xes.in.XesXmlParser;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XLog;
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
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserFactory;

import lpsolve.LpSolve;
import nl.tue.alignment.Progress;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.ReplayerParameters;
import nl.tue.alignment.Utils;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;

public class AlignmentTest {

	private static final int THREADS = 1;//Math.max(1, Runtime.getRuntime().availableProcessors() / 2);

	private static String FOLDER = "c:/temp/alignment/";
	private static String SEP = Utils.SEP;
	public static int iteration = 0;
	public static FrameContext frame = new FrameContext();

	static String[] SINGLETRACE = new String[] {};

	public static enum Type {
		DIJKSTRA(false), //
		ASTAR(false), //
		ASTARRED(false), //
		INC0(false), //
		INC0RED(true), //
		INC3(false), //
		INC10(false), //
		INC_PLUS(false), //
		PLANNING(false);

		private boolean include;

		private Type(boolean include) {
			this.include = include;
		}

		public boolean include() {
			return include;
		}
	}

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

		if (Type.PLANNING.include()) {
			frame.setVisible(true);
		}
		mainFileFolder(Debug.STATS, Integer.MAX_VALUE, "CCC19-complete", "CCC19");
		System.exit(0);
		mainFileFolder(Debug.STATS, Integer.MAX_VALUE, "software");

		//		mainFileFolder(Debug.STATS, "bpi12");//"pr1151_l4_noise","pr1912_l4_noise");
		//		mainFileFolder(Debug.STATS, "test");//"pr1151_l4_noise","pr1912_l4_noise");
		//		mainFileFolder(Debug.STATS, "pr1151_l4_noise", "pr1912_l4_noise", "temp", "sepsis", "prCm6", "prDm6", "prEm6",
		//				"prFm6", "prGm6", "prAm6", "prBm6");
		//		mainFileFolder(Debug.STATS, 15, "road_fines");

		//		mainFileFolder(Debug.STATS, 60, "pr1151_l4_noise", "pr1912_l4_noise");

		//		mainFileFolder(SINGLETRACE.length > 0 ? Debug.DOT : Debug.STATS, 100000, "log_abc_500");
		//		mainFileFolder(SINGLETRACE.length > 0 ? Debug.DOT : Debug.STATS, 100000, "loopdouble_500K");
		//
		//		System.exit(0);

		//Initialize internal structures...

		//		mainFileFolder(Debug.NONE, 100000, "test", "test2", "alifah", "alifah2");

		//		mainFileFolder(Debug.DOT, Integer.MAX_VALUE, "alifah3");

		// mainFileFolder(Debug.STATS, Integer.MAX_VALUE, "BPIC15_1_start_end_IMfa");
		//		mainFileFolder(Debug.STATS, 30, "pr1151_l4_noise");
		//		mainFileFolder(Debug.STATS, 15, "prCm6");

		//		mainFileFolder(Debug.STATS, 30, "prAm6", "prBm6", "prEm6", "prCm6", "prDm6", "prFm6", "prGm6");

		//April 2018:
		int timeout = 10;
		//
		mainFileFolder(Debug.STATS, timeout, "prCm6", "prAm6", "prEm6", "prBm6", "prFm6", "prGm6", "prDm6"); // Planner runs out of memory
		//		mainFileFolder(Debug.STATS, timeout, "bpi12");
		//		mainFileFolder(Debug.STATS, timeout, "pr1151_l4_noise", "pr1912_l4_noise");
		//
		//		mainFolder(Debug.NONE, timeout, "laura/");//
		//		mainFolder(Debug.NONE, timeout, "isbpm2013/");
		//
		//		mainFileFolder(Debug.NONE, timeout, "d53_rad1", "d62_rad1", "d63_rad1", "d64_rad1", //
		//				"d53_rad1_10_noise", "d62_rad1_10_noise");
		//		mainFileFolder(Debug.NONE, timeout, "test", "d63_rad1_10_noise", "d64_rad1_10_noise", //
		//				"d53_rad1_20_noise", "d62_rad1_20_noise", "d63_rad1_20_noise", "d64_rad1_20_noise", //
		//				"d53_rad1_30_noise", "d62_rad1_30_noise", "d63_rad1_30_noise", "d64_rad1_30_noise");

		if (Type.PLANNING.include()) {
			frame.setVisible(false);
		}

	}

	public static void mainFolder(Debug debug, int timeoutSecondsPerTrace, String... eval) throws Exception {

		System.out.print("filename" + SEP + "logsize" + SEP);
		for (Type type : Type.values()) {
			System.out.print(type + " number timeout" + SEP);
			System.out.print(type + " runtime (ms)" + SEP);
			System.out.print(type + " CPU time (ms)" + SEP);
			System.out.print(type + " preprocess time (ms)" + SEP);
			System.out.print(type + " memory (kb)" + SEP);
			System.out.print(type + " solved LPs" + SEP);
			System.out.print(type + " cost" + SEP);
		}
		System.out.println();

		for (String folder : eval) {

			String[] names = new File(FOLDER + folder).list(new FilenameFilter() {

				public boolean accept(File dir, String name) {
					return name.endsWith(".pnml");
				}
			});

			for (String name : names) {
				name = name.replace(".pnml", "");

				File pnmlFile = new File(FOLDER + folder + name + ".pnml");
				FileInputStream inputStream = new FileInputStream(pnmlFile);
				Pnml pnml = importPnmlFromStream(inputStream, pnmlFile.getName(), pnmlFile.length());
				inputStream.close();

				Petrinet net = PetrinetFactory.newPetrinet(pnml.getLabel());
				Marking initialMarking = new Marking();
				GraphLayoutConnection layout = new GraphLayoutConnection(net);

				pnml.convertToNet(net, initialMarking, layout);

				//				Petrinet net = constructNet(FOLDER + folder + name + ".pnml");
				//				Marking initialMarking = getInitialMarking(net);
				Marking finalMarking = getFinalMarking(net);
				XLog log;
				XEventClassifier eventClassifier;

				XMxmlParser parser = new XMxmlParser();
				eventClassifier = XLogInfoImpl.STANDARD_CLASSIFIER;
				log = parser.parse(new File(FOLDER + folder + name + ".xml")).get(0);

				String f = FOLDER + folder + name;
				System.out.print(f + SEP);
				System.out.print((log.size() + 1) + SEP);
				for (Type type : Type.values()) {
					try {
						doReplayExperiment(debug, f, net, initialMarking, finalMarking, log, eventClassifier, type,
								timeoutSecondsPerTrace);
					} catch (Exception e) {
						System.err.println(e.getMessage());
						e.printStackTrace();
					}

				}
				System.out.println();
			}
		}
	}

	public static void mainFileFolder(Debug debug, int timeoutSecondsPerTrace, String... names) throws Exception {

		System.out.print("filename" + SEP + "logsize" + SEP);
		for (Type type : Type.values()) {
			if (type == Type.PLANNING && type.include()) {
				System.out.print("PL preCPU" + SEP + "PL preClock" + SEP + "PL planCPU" + SEP + "PL planClock" + SEP
						+ "PL ParseClock" + SEP + "PL TotalClock" + SEP);

			} else {
				if (type.include()) {
					System.out.print(type + " number timeout" + SEP);
					System.out.print(type + " runtime (ms)" + SEP);
					System.out.print(type + " CPU time (ms)" + SEP);
					System.out.print(type + " preprocess time (ms)" + SEP);
					System.out.print(type + " memory (kb)" + SEP);
					System.out.print(type + " solved LPs" + SEP);
					System.out.print(type + " cost" + SEP);
				}
			}

		}
		System.out.println();

		for (String name : names) {
			String folder = FOLDER + name + "/" + name;

			File pnmlFile = new File(folder + ".pnml");
			FileInputStream inputStream = new FileInputStream(pnmlFile);
			Pnml pnml = importPnmlFromStream(inputStream, pnmlFile.getName(), pnmlFile.length());
			inputStream.close();

			Petrinet net = PetrinetFactory.newPetrinet(pnml.getLabel());
			Marking initialMarking = new Marking();
			GraphLayoutConnection layout = new GraphLayoutConnection(net);

			pnml.convertToNet(net, initialMarking, layout);

			//			Petrinet net = constructNet(folder + ".pnml");
			//			Marking initialMarking = getInitialMarking(net);
			Marking finalMarking = getFinalMarking(net);
			XLog log;
			XEventClassifier eventClassifier;

			if (new File(folder + ".mxml").exists()) {
				XMxmlParser parser = new XMxmlParser();
				eventClassifier = XLogInfoImpl.STANDARD_CLASSIFIER;
				log = parser.parse(new File(folder + ".mxml")).get(0);
			} else {
				XesXmlParser parser = new XesXmlParser();

				Iterator<Transition> it = net.getTransitions().iterator();
				Transition t;
				do {
					t = it.next();

				} while (t.isInvisible() && it.hasNext());

				if (t.getLabel().contains("+")) {
					eventClassifier = XLogInfoImpl.STANDARD_CLASSIFIER;
				} else {
					eventClassifier = new XEventNameClassifier();
				}
				log = parser.parse(new File(folder + ".xes")).get(0);
			}

			//			for (XTrace trace : log) {
			//				for (XEvent event : trace) {
			//					long time = XTimeExtension.instance().extractTimestamp(event).getTime();
			//					if (XLifecycleExtension.instance().extractTransition(event).equals("complete")) {
			//						XTimeExtension.instance().assignTimestamp(event, time + 1);
			//					}
			//				}
			//				trace.sort(new Comparator<XEvent>() {
			//
			//					public int compare(XEvent o1, XEvent o2) {
			//						return XTimeExtension.instance().extractTimestamp(o1)
			//								.compareTo(XTimeExtension.instance().extractTimestamp(o2));
			//					}
			//				});
			//			}

			//			for (int i = log.size(); i-- > 20;) {
			//				log.remove(i);
			//			}
			///////////////
			//			java.util.Iterator<XTrace> it = log.iterator();
			//			while (it.hasNext()) {
			//				if (!XConceptExtension.instance().extractName(it.next()).equals("1547915248799-video_1.g_CVC")) {
			//					it.remove();
			//				}
			//			}
			///////////////

			System.out.print(folder + SEP);
			System.out.print((log.size() + 1) + SEP);
			for (Type type : Type.values()) {
				try {
					doReplayExperiment(debug, folder, net, initialMarking, finalMarking, log, eventClassifier, type,
							timeoutSecondsPerTrace);
				} catch (Exception e) {
					System.err.println("Exception: " + e.getMessage());
					e.printStackTrace();
				}

			}
			System.out.flush();
			System.out.println();
		}
		System.exit(0);
	}

	private static void doReplayExperiment(Debug debug, String folder, Petrinet net, Marking initialMarking,
			Marking finalMarking, XLog log, XEventClassifier eventClassifier, Type type, int timeoutPerTraceInSec)
			throws FileNotFoundException, InterruptedException, ExecutionException {

		XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
		TransEvClassMapping mapping = constructMapping(net, log, dummyEvClass, eventClassifier);
		XLogInfo summary = XLogInfoFactory.createLogInfo(log, eventClassifier);
		XEventClasses classes = summary.getEventClasses();

		int threads;
		if (debug == Debug.STATS) {
			threads = THREADS;
		} else if (debug == Debug.DOT) {
			threads = 1;
		} else {
			threads = THREADS;
		}

		// timeout  per trace 
		int timeout = log.size() * timeoutPerTraceInSec * 1000 / 10;
		int maxNumberOfStates = Integer.MAX_VALUE;

		boolean moveSort = false;
		boolean useInt = false;
		boolean partialOrder = true;
		boolean preferExact = true;
		boolean queueSort = true;
		ReplayerParameters parameters;
		boolean preProcessUsingPlaceBasedConstraints = true;
		int maxReducedSequenceLength = 1;

		switch (type) {
			case DIJKSTRA :
				if (type.include()) {
					parameters = new ReplayerParameters.Dijkstra(moveSort, queueSort, threads, debug, timeout,
							maxNumberOfStates, Integer.MAX_VALUE, partialOrder);
					doReplay(debug, folder, "Dijkstra", net, initialMarking, finalMarking, log, mapping, classes,
							parameters);
				}
				break;

			case ASTAR :
				if (type.include()) {
					parameters = new ReplayerParameters.AStar(moveSort, queueSort, preferExact, threads, useInt, debug,
							timeout, maxNumberOfStates, Integer.MAX_VALUE, partialOrder);
					doReplay(debug, folder, "AStar", net, initialMarking, finalMarking, log, mapping, classes,
							parameters);
				}
				break;
			case ASTARRED :
				if (type.include()) {
					parameters = new ReplayerParameters.AStar(moveSort, queueSort, preferExact, threads, useInt, debug,
							timeout, maxNumberOfStates, Integer.MAX_VALUE, partialOrder, maxReducedSequenceLength);
					doReplay(debug, folder, "AStarReduced-" + maxReducedSequenceLength, net, initialMarking,
							finalMarking, log, mapping, classes, parameters);
				}
				break;

			case INC0 :
				if (type.include()) {
					parameters = new ReplayerParameters.IncrementalAStar(moveSort, threads, useInt, debug, timeout,
							maxNumberOfStates, Integer.MAX_VALUE, partialOrder, 0);
					doReplay(debug, folder, "Incre0", net, initialMarking, finalMarking, log, mapping, classes,
							parameters);
				}
				break;
			case INC0RED :
				if (type.include()) {
					parameters = new ReplayerParameters.IncrementalAStar(moveSort, threads, useInt, debug, timeout,
							maxNumberOfStates, Integer.MAX_VALUE, partialOrder, false, 0, maxReducedSequenceLength);
					doReplay(debug, folder, "Incre0Reduced" + maxReducedSequenceLength, net, initialMarking,
							finalMarking, log, mapping, classes, parameters);
				}
				break;
			case INC3 :
				if (type.include()) {
					parameters = new ReplayerParameters.IncrementalAStar(moveSort, threads, useInt, debug, timeout,
							maxNumberOfStates, Integer.MAX_VALUE, partialOrder, 3);
					doReplay(debug, folder, "Incre3", net, initialMarking, finalMarking, log, mapping, classes,
							parameters);
				}
				break;
			case INC10 :
				if (type.include()) {
					parameters = new ReplayerParameters.IncrementalAStar(moveSort, threads, useInt, debug, timeout,
							maxNumberOfStates, Integer.MAX_VALUE, partialOrder, 10);
					doReplay(debug, folder, "Incre10", net, initialMarking, finalMarking, log, mapping, classes,
							parameters);
				}
				break;

			case INC_PLUS :
				if (type.include()) {
					parameters = new ReplayerParameters.IncrementalAStar(moveSort, threads, useInt, debug, timeout,
							maxNumberOfStates, Integer.MAX_VALUE, partialOrder, true);
					doReplay(debug, folder, "Incre++", net, initialMarking, finalMarking, log, mapping, classes,
							parameters);
				}
				break;

			case PLANNING :
				//				if (type.include()) {
				//					PlanningBasedAlignmentParameters planParameters = new PlanningBasedAlignmentParameters();
				//					planParameters.setInitialMarking(initialMarking);
				//					planParameters.setFinalMarking(finalMarking);
				//					Map<XEventClass, Integer> movesOnLogCosts = new HashMap<>();
				//					for (XEventClass ec : classes.getClasses()) {
				//						movesOnLogCosts.put(ec, 1);
				//					}
				//
				//					// experiments use default costs
				//					planParameters.setMovesOnLogCosts(movesOnLogCosts);
				//					Map<Transition, Integer> movesOnModelCosts = new HashMap<>();
				//					Map<Transition, Integer> synchronousMovesCosts = new HashMap<>();
				//					for (Transition t : net.getTransitions()) {
				//						movesOnModelCosts.put(t, t.isInvisible() ? new Integer(0) : new Integer(1));
				//						synchronousMovesCosts.put(t, 0);
				//					}
				//
				//					planParameters.setMovesOnModelCosts(movesOnModelCosts);
				//					planParameters.setSynchronousMovesCosts(synchronousMovesCosts);
				//					planParameters.setPlannerSearchStrategy(PlannerSearchStrategy.BLIND_A_STAR);
				//					planParameters.setTracesInterval(new int[] { 1, log.size() });
				//					planParameters.setTracesLengthBounds(new int[] { 0, Integer.MAX_VALUE });
				//
				//					planParameters.setTransitionsEventsMapping(mapping);
				//					planParameters.setPartiallyOrderedEvents(false);
				//
				//					doReplayPlanning(debug, folder, "Planning", net, initialMarking, finalMarking, log, mapping,
				//							classes, planParameters);
				//				}
				break;

		}
	}

	private static void doReplay(Debug debug, String folder, String postfix, PetrinetGraph net, Marking initialMarking,
			Marking finalMarking, XLog log, TransEvClassMapping mapping, XEventClasses classes,
			ReplayerParameters parameters) throws FileNotFoundException, InterruptedException, ExecutionException {
		PrintStream stream;
		if (debug == Debug.STATS) {
			stream = new PrintStream(new File(folder + " " + postfix + ".csv"));
		} else if (debug == Debug.DOT) {
			stream = new PrintStream(new File(folder + "_" + postfix + ".dot"));
		} else {
			stream = System.out;
		}
		ReplayAlgorithm.Debug.setOutputStream(stream);

		long start = System.nanoTime();
		Replayer replayer = new Replayer(parameters, (Petrinet) net, initialMarking, finalMarking, classes, mapping,
				true);

		PNRepResult result = replayer.computePNRepResult(Progress.INVISIBLE, log);//, SINGLETRACE);
		long end = System.nanoTime();

		int cost = (int) Double.parseDouble((String) result.getInfo().get(Replayer.MAXMODELMOVECOST));
		int timeout = 0;
		double time = 0;
		int mem = 0;
		int lps = 0;
		double pretime = 0;
		for (SyncReplayResult res : result) {
			cost += res.getTraceIndex().size() * res.getInfo().get(PNRepResult.RAWFITNESSCOST);
			timeout += res.getTraceIndex().size() * (res.getInfo().get(Replayer.TRACEEXITCODE).intValue() != 1 ? 1 : 0);
			time += res.getInfo().get(PNRepResult.TIME);
			pretime += res.getInfo().get(Replayer.PREPROCESSTIME);
			lps += res.getInfo().get(Replayer.HEURISTICSCOMPUTED);
			mem = Math.max(mem, res.getInfo().get(Replayer.MEMORYUSED).intValue());
		}

		if (stream != System.out) {
			//			System.out.println(result.getInfo().toString());
			stream.close();
		}

		// number timeouts
		System.out.print(timeout + SEP);
		// clocktime
		System.out.print(String.format("%.3f", (end - start) / 1000000.0) + SEP);
		// cpu time
		System.out.print(String.format("%.3f", time) + SEP);
		// preprocess time
		System.out.print(String.format("%.3f", pretime) + SEP);
		// max memory
		System.out.print(mem + SEP);
		// solves lps.
		System.out.print(lps + SEP);
		// total cost.
		System.out.print(cost + SEP);

		System.out.flush();

	}

	//	private static void doReplayPlanning(Debug debug, String folder, String postfix, Petrinet net,
	//			Marking initialMarking, Marking finalMarking, XLog log, TransEvClassMapping mapping, XEventClasses classes,
	//			PlanningBasedAlignmentParameters parameters)
	//			throws FileNotFoundException, InterruptedException, ExecutionException {
	//
	//		PrintStream stream;
	//		if (debug == Debug.STATS) {
	//			stream = new PrintStream(new File(folder + " " + postfix + ".csv"));
	//		} else if (debug == Debug.DOT) {
	//			stream = new PrintStream(new File(folder + "_" + postfix + ".dot"));
	//		} else {
	//			stream = System.out;
	//		}
	//		ReplayAlgorithm.Debug.setOutputStream(stream);
	//
	//		long start = System.nanoTime();
	//
	//		PlanningBasedAlignmentPlugin plugin = new PlanningBasedAlignmentPlugin();
	//		PlanningBasedReplayResult result = plugin.align(frame, new File("E:/"), log, net, parameters);
	//
	//		long end = System.nanoTime();
	//
	//		//		int cost = (int) Double.parseDouble((String) result.getInfo().get(Replayer.MAXMODELMOVECOST));
	//		//		int timeout = 0;
	//		//		double time = 0;
	//		//		int mem = 0;
	//		//		double pretime = 0;
	//		//		for (SyncReplayResult res : result) {
	//		//			cost += res.getTraceIndex().size() * res.getInfo().get(PNRepResult.RAWFITNESSCOST);
	//		//			timeout += res.getTraceIndex().size() * (res.getInfo().get(Replayer.TRACEEXITCODE).intValue() != 1 ? 1 : 0);
	//		//			time += res.getInfo().get(PNRepResult.TIME);
	//		//			pretime += res.getInfo().get(Replayer.PREPROCESSTIME);
	//		//			mem = Math.max(mem, res.getInfo().get(Replayer.MEMORYUSED).intValue());
	//		//		}
	//		//
	//		//		if (stream != System.out) {
	//		//			//			System.out.println(result.getInfo().toString());
	//		//			stream.close();
	//		//		}
	//		//
	//		//		// number timeouts
	//		//		System.out.print(timeout + SEP);
	//		// clocktime
	//		//		System.out.print(String.format("%.3f", (end - start) / 1000000.0) + SEP);
	//		//		// cpu time
	//		//		System.out.print(String.format("%.3f", time) + SEP);
	//		//		// preprocess time
	//		//		System.out.print(String.format("%.3f", pretime) + SEP);
	//		//		// max memory
	//		//		System.out.print(mem + SEP);
	//		//		// total cost.
	//		//		System.out.print(cost + SEP);
	//		//
	//		System.out.flush();
	//
	//	}

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

	//	private static Petrinet constructNet(String netFile) {
	//		PNMLSerializer PNML = new PNMLSerializer();
	//		NetSystem sys = PNML.parse(netFile);
	//
	//		//System.err.println(sys.getMarkedPlaces());
	//
	//		//		int pi, ti;
	//		//		pi = ti = 1;
	//		//		for (org.jbpt.petri.Place p : sys.getPlaces())
	//		//			p.setName("p" + pi++);
	//		//		for (org.jbpt.petri.Transition t : sys.getTransitions())
	//		//				t.setName("t" + ti++);
	//
	//		Petrinet net = PetrinetFactory.newPetrinet(netFile);
	//
	//		// places
	//		Map<org.jbpt.petri.Place, Place> p2p = new HashMap<org.jbpt.petri.Place, Place>();
	//		for (org.jbpt.petri.Place p : sys.getPlaces()) {
	//			Place pp = net.addPlace(p.toString());
	//			p2p.put(p, pp);
	//		}
	//
	//		// transitions
	//		Map<org.jbpt.petri.Transition, Transition> t2t = new HashMap<org.jbpt.petri.Transition, Transition>();
	//		for (org.jbpt.petri.Transition t : sys.getTransitions()) {
	//			Transition tt = net.addTransition(t.getLabel());
	//			if (t.isSilent() || t.getLabel().startsWith("tau") || t.getLabel().equals("t2") || t.getLabel().equals("t8")
	//					|| t.getLabel().equals("complete")) {
	//				tt.setInvisible(true);
	//			}
	//			t2t.put(t, tt);
	//		}
	//
	//		// flow
	//		for (Flow f : sys.getFlow()) {
	//			if (f.getSource() instanceof org.jbpt.petri.Place) {
	//				net.addArc(p2p.get(f.getSource()), t2t.get(f.getTarget()));
	//			} else {
	//				net.addArc(t2t.get(f.getSource()), p2p.get(f.getTarget()));
	//			}
	//		}
	//
	//		// add unique start node
	//		if (sys.getSourceNodes().isEmpty()) {
	//			Place i = net.addPlace("START_P");
	//			Transition t = net.addTransition("");
	//			t.setInvisible(true);
	//			net.addArc(i, t);
	//
	//			for (org.jbpt.petri.Place p : sys.getMarkedPlaces()) {
	//				net.addArc(t, p2p.get(p));
	//			}
	//
	//		}
	//
	//		return net;
	//	}

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
