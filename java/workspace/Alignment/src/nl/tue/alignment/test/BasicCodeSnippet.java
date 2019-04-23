package nl.tue.alignment.test;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.classification.XEventNameClassifier;
import org.deckfour.xes.in.XUniversalParser;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XLog;
import org.jbpt.petri.Flow;
import org.jbpt.petri.NetSystem;
import org.jbpt.petri.io.PNMLSerializer;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetGraph;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.graphbased.directed.petrinet.impl.PetrinetFactory;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.petrinet.replayresult.StepTypes;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import lpsolve.LpSolve;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.ReplayerParameters;
import nl.tue.alignment.TraceReplayTask;
import nl.tue.alignment.Utils;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;

public class BasicCodeSnippet {

	public static void main(String[] args) throws Exception {
		// INITIALIZE LpSolve for stdout
		LpSolve.lpSolveVersion();

		String petrinetFile = "C:\\temp\\dot\\test2.pnml";
		String logFile = "C:\\temp\\dot\\test2.mxml";

		Petrinet net = constructNet(petrinetFile);
		Marking initialMarking = getInitialMarking(net);
		Marking finalMarking = getFinalMarking(net);

		XLog log;
		XEventClassifier eventClassifier;

		log = new XUniversalParser().parse(new File(logFile)).iterator().next();

		// matching using A (typical for xes files)
		eventClassifier = new XEventNameClassifier();

		Iterator<Transition> it = net.getTransitions().iterator();
		while (it.hasNext()) {
			Transition t = it.next();
			if (!t.isInvisible() && t.getLabel().endsWith("+complete")) {
				// matching using A+Complete (typical for mxml files)
				eventClassifier = XLogInfoImpl.STANDARD_CLASSIFIER;
				break;
			}
		}

		XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
		TransEvClassMapping mapping = constructMappingBasedOnLabelEquality(net, log, dummyEvClass, eventClassifier);
		XLogInfo summary = XLogInfoFactory.createLogInfo(log, eventClassifier);
		XEventClasses classes = summary.getEventClasses();

		Map<Transition, Integer> costModelMove = new HashMap<>();
		Map<Transition, Integer> costSyncMove = new HashMap<>();
		Map<XEventClass, Integer> costLogMove = new HashMap<>();
		for (Transition t : net.getTransitions()) {
			costSyncMove.put(t, 0);
			costModelMove.put(t, t.isInvisible() ? 0 : 2);
		}
		for (XEventClass c : summary.getEventClasses().getClasses()) {
			costLogMove.put(c, 5);
		}
		costLogMove.put(dummyEvClass, 5);

		System.out.println(String.format("Log size: %d events, %d traces, %d classes", summary.getNumberOfEvents(),
				summary.getNumberOfTraces(), (summary.getEventClasses().size() + 1)));
		System.out.println(String.format("Model size: %d transitions, %d places", net.getTransitions().size(),
				net.getPlaces().size()));

		doReplay(log, net, initialMarking, finalMarking, classes, mapping, costModelMove, costSyncMove, costLogMove);

	}

	public static void doReplay(XLog log, Petrinet net, Marking initialMarking, Marking finalMarking,
			XEventClasses classes, TransEvClassMapping mapping, Map<Transition, Integer> costModelMove,
			Map<Transition, Integer> costSyncMove, Map<XEventClass, Integer> costLogMove) {

		int nThreads = 2;
		int costUpperBound = Integer.MAX_VALUE;
		// timeout per trace in milliseconds
		int timeoutMilliseconds = 10 * 1000;

		int maximumNumberOfStates = Integer.MAX_VALUE;
		ReplayerParameters parameters;

		parameters = new ReplayerParameters.Dijkstra(false, false, nThreads, Debug.DOT, timeoutMilliseconds,
				maximumNumberOfStates, costUpperBound, false, 2, true);

		//Current: 
		//		parameters = new ReplayerParameters.IncrementalAStar(false, nThreads, false, Debug.DOT, timeoutMilliseconds,
		//				maximumNumberOfStates, costUpperBound, false, false, 0, 3);

		//		//BPM2018: 
		//		parameters = new ReplayerParameters.IncrementalAStar(false, nThreads, false, Debug.DOT,
		//						timeoutMilliseconds, maximumNumberOfStates, costUpperBound, false, false);
		//Traditional
		//		parameters = new ReplayerParameters.AStar(true, true, true, nThreads, true, Debug.NONE,
		//				timeoutMilliseconds, maximumNumberOfStates, costUpperBound, false);

		Replayer replayer = new Replayer(parameters, net, initialMarking, finalMarking, classes, costModelMove,
				costLogMove, costSyncMove, mapping, false);

		// preprocessing time to be added to the statistics if necessary
		long preProcessTimeNanoseconds = 0;

		int success = 0;
		int failed = 0;
		ExecutorService service = Executors.newFixedThreadPool(parameters.nThreads);

		@SuppressWarnings("unchecked")
		Future<TraceReplayTask>[] futures = new Future[log.size()];

		for (int i = 0; i < log.size(); i++) {
			// Setup the trace replay task
			TraceReplayTask task = new TraceReplayTask(replayer, parameters, log.get(i), i, timeoutMilliseconds,
					parameters.maximumNumberOfStates, preProcessTimeNanoseconds);

			// submit for execution
			futures[i] = service.submit(task);
		}
		// initiate shutdown and wait for termination of all submitted tasks.
		service.shutdown();

		// obtain the results one by one.

		for (int i = 0; i < log.size(); i++) {

			TraceReplayTask result;
			try {
				result = futures[i].get();
			} catch (Exception e) {
				// execution os the service has terminated.
				throw new RuntimeException("Error while executing replayer in ExecutorService. Interrupted maybe?", e);
			}
			switch (result.getResult()) {
				case DUPLICATE :
					assert false; // cannot happen in this setting
					throw new RuntimeException("Result cannot be a duplicate in per-trace computations.");
				case FAILED :
					// internal error in the construction of synchronous product or other error.
					throw new RuntimeException("Error in alignment computations");
				case SUCCESS :
					// process succcesful execution of the replayer
					SyncReplayResult replayResult = result.getSuccesfulResult();
					int exitCode = replayResult.getInfo().get(Replayer.TRACEEXITCODE).intValue();
					if ((exitCode & Utils.OPTIMALALIGNMENT) == Utils.OPTIMALALIGNMENT) {
						// Optimal alignment found.
						success++;

						System.out.println(String.format("Time (ms): %f",
								result.getSuccesfulResult().getInfo().get(PNRepResult.TIME)));
						//			System.out.println(result.getSuccesfulResult().getStepTypes());

						int logMove = 0, syncMove = 0, modelMove = 0, tauMove = 0;
						for (StepTypes step : result.getSuccesfulResult().getStepTypes()) {
							if (step == StepTypes.L) {
								logMove++;
							} else if (step == StepTypes.LMGOOD) {
								syncMove++;
							} else if (step == StepTypes.MREAL) {
								modelMove++;
							} else if (step == StepTypes.MINVI) {
								tauMove++;
							}
						}
						System.out.println(String.format("Log %d, Model %d, Sync %d, tau %d", logMove, modelMove,
								syncMove, tauMove));

					} else if ((exitCode & Utils.FAILEDALIGNMENT) == Utils.FAILEDALIGNMENT) {
						// failure in the alignment. Error code shows more details.
						failed++;
					}
					if ((exitCode & Utils.ENABLINGBLOCKEDBYOUTPUT) == Utils.ENABLINGBLOCKEDBYOUTPUT) {
						// in some marking, there were too many tokens in a place, blocking the addition of more tokens. Current upper limit is 128
					}
					if ((exitCode & Utils.COSTFUNCTIONOVERFLOW) == Utils.COSTFUNCTIONOVERFLOW) {
						// in some marking, the cost function went through the upper limit of 2^24
					}
					if ((exitCode & Utils.HEURISTICFUNCTIONOVERFLOW) == Utils.HEURISTICFUNCTIONOVERFLOW) {
						// in some marking, the heuristic function went through the upper limit of 2^24
					}
					if ((exitCode & Utils.TIMEOUTREACHED) == Utils.TIMEOUTREACHED) {
						// alignment failed with a timeout
					}
					if ((exitCode & Utils.STATELIMITREACHED) == Utils.STATELIMITREACHED) {
						// alignment failed due to reacing too many states.
					}
					if ((exitCode & Utils.COSTLIMITREACHED) == Utils.COSTLIMITREACHED) {
						// no optimal alignment found with cost less or equal to the given limit.
					}
					if ((exitCode & Utils.CANCELLED) == Utils.CANCELLED) {
						// user-cancelled.
					}
					if ((exitCode & Utils.FINALMARKINGUNREACHABLE) == Utils.FINALMARKINGUNREACHABLE) {
						// user-cancelled.
						System.err.println("final marking unreachable.");
					}

					break;
			}
		}
		System.out.println("Successful: " + success);
		System.out.println("Failed:     " + failed);

	}

	public static TransEvClassMapping constructMappingBasedOnLabelEquality(PetrinetGraph net, XLog log,
			XEventClass dummyEvClass, XEventClassifier eventClassifier) {
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
				}
			}

			if (!mapped && !t.isInvisible()) {
				mapping.put(t, dummyEvClass);
			}

		}

		return mapping;
	}

	public static Petrinet constructNet(String netFile) {
		PNMLSerializer PNML = new PNMLSerializer();
		NetSystem sys = PNML.parse(netFile);

		//System.err.println(sys.getMarkedPlaces());

		//		int pi, ti;
		//		pi = ti = 1;
		//		for (org.jbpt.petri.Place p : sys.getPlaces())
		//			p.setName("p" + pi++);
		//		for (org.jbpt.petri.Transition t : sys.getTransitions())
		//				t.setName("t" + ti++);

		Petrinet net = PetrinetFactory.newPetrinet(netFile);

		// places
		Map<org.jbpt.petri.Place, Place> p2p = new HashMap<org.jbpt.petri.Place, Place>();
		for (org.jbpt.petri.Place p : sys.getPlaces()) {
			Place pp = net.addPlace(p.toString());
			p2p.put(p, pp);
		}

		// transitions
		Map<org.jbpt.petri.Transition, Transition> t2t = new HashMap<org.jbpt.petri.Transition, Transition>();
		for (org.jbpt.petri.Transition t : sys.getTransitions()) {
			Transition tt = net.addTransition(t.getLabel());
			if (t.isSilent() || t.getLabel().startsWith("tau") || t.getLabel().equals("t2") || t.getLabel().equals("t8")
					|| t.getLabel().equals("complete")) {
				tt.setInvisible(true);
			}
			t2t.put(t, tt);
		}

		// flow
		for (Flow f : sys.getFlow()) {
			if (f.getSource() instanceof org.jbpt.petri.Place) {
				net.addArc(p2p.get(f.getSource()), t2t.get(f.getTarget()));
			} else {
				net.addArc(t2t.get(f.getSource()), p2p.get(f.getTarget()));
			}
		}

		// add unique start node
		if (sys.getSourceNodes().isEmpty()) {
			Place i = net.addPlace("START_P");
			Transition t = net.addTransition("");
			t.setInvisible(true);
			net.addArc(i, t);

			for (org.jbpt.petri.Place p : sys.getMarkedPlaces()) {
				net.addArc(t, p2p.get(p));
			}

		}

		return net;
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

}
