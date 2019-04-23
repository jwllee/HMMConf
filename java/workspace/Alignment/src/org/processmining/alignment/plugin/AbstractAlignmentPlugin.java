package org.processmining.alignment.plugin;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.model.XLog;
import org.processmining.framework.plugin.PluginContext;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetGraph;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayer.algorithms.IPNReplayAlgorithm;
import org.processmining.plugins.petrinet.replayer.algorithms.IPNReplayParamProvider;
import org.processmining.plugins.petrinet.replayer.algorithms.IPNReplayParameter;
import org.processmining.plugins.petrinet.replayer.algorithms.costbasedcomplete.CostBasedCompleteParam;
import org.processmining.plugins.petrinet.replayer.algorithms.costbasedcomplete.CostBasedCompleteParamProvider;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;

import nl.tue.alignment.Progress;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.ReplayerParameters;
import nl.tue.astar.AStarException;

public abstract class AbstractAlignmentPlugin implements IPNReplayAlgorithm {

	private Map<Transition, Integer> mapTrans2Cost;
	private Map<XEventClass, Integer> mapEvClass2Cost;
	private Map<Transition, Integer> mapSync2Cost;
	private Marking initMarking;
	private Marking[] finalMarkings;
	private int numThreads;

	protected AbstractAlignmentPlugin() {
	}

	public PNRepResult replayLog(final PluginContext context, PetrinetGraph net, XLog xLog, TransEvClassMapping mapping,
			IPNReplayParameter parameters) throws AStarException {

		importParameters((CostBasedCompleteParam) parameters);

		context.getProgress().setMaximum(xLog.size() + 1);

		ReplayerParameters replayParameters = constructReplayParameters(numThreads,
				((CostBasedCompleteParam) parameters).isPartiallyOrderedEvents(),
				((CostBasedCompleteParam) parameters).getMaxNumOfStates());

		XLogInfo summary = XLogInfoFactory.createLogInfo(xLog, mapping.getEventClassifier());
		Replayer replayer = constructReplayer((Petrinet) net, xLog, mapping, replayParameters, summary);

		PNRepResult result;
		try {
			result = replayer.computePNRepResult(new Progress() {

				public void setMaximum(int maximum) {
					context.getProgress().setMinimum(0);
					context.getProgress().setMaximum(maximum);
				}

				public void inc() {
					context.getProgress().inc();
				}

				public boolean isCanceled() {
					return context.getProgress().isCancelled();
				}

				public void log(String message) {
					context.log(message);
				}
			}, xLog);
		} catch (InterruptedException | ExecutionException e) {
			throw new AStarException(e);
		}

		result.addInfo(PNRepResult.VISTITLE,
				"Alignments of " + XConceptExtension.instance().extractName(xLog) + " on " + net.getLabel());

		return result;

	}

	protected Replayer constructReplayer(Petrinet net, XLog xLog, TransEvClassMapping mapping,
			ReplayerParameters replayParameters, XLogInfo summary) {
		Replayer replayer = new Replayer(replayParameters, net, initMarking, finalMarkings[0],
				summary.getEventClasses(), mapTrans2Cost, mapEvClass2Cost, mapSync2Cost, mapping, true);
		return replayer;
	}

	protected abstract ReplayerParameters constructReplayParameters(int numThreads, boolean usePartialOrder,
			int maximumNumberOfStates);

	public IPNReplayParamProvider constructParamProvider(PluginContext context, PetrinetGraph net, XLog log,
			TransEvClassMapping mapping) {
		return new CostBasedCompleteParamProvider(context, net, log, mapping);
	}

	/**
	 * Return true if all replay inputs are correct
	 */
	public boolean isAllReqSatisfied(PluginContext context, PetrinetGraph net, XLog log, TransEvClassMapping mapping,
			IPNReplayParameter parameter) {

		if (isReqWOParameterSatisfied(context, net, log, mapping)) {
			if (isParameterReqCorrect(net, log, mapping, parameter)) {
				Marking[] finalMarking = ((CostBasedCompleteParam) parameter).getFinalMarkings();
				if ((finalMarking != null) && (finalMarking.length == 1)) {
					return true;
				}
			}
		}

		return false;
	}

	/**
	 * Return true if input of replay without parameters are correct
	 */
	public boolean isReqWOParameterSatisfied(PluginContext context, PetrinetGraph net, XLog log,
			TransEvClassMapping mapping) {
		//		if ((net instanceof ResetInhibitorNet) || (net instanceof InhibitorNet) || (net instanceof ResetNet)
		//				|| (net instanceof Petrinet) || (net instanceof OpenNet)) {
		if ((net instanceof Petrinet)) {
			// check number of transitions, places, and event classes, should be less than Short.MAX_VALUE
			if ((net.getTransitions().size() < Short.MAX_VALUE) && (net.getPlaces().size() < Short.MAX_VALUE)) {
				// check the number of event classes, should be less than Short.MAX_VALUE
				XLogInfo summary = XLogInfoFactory.createLogInfo(log, mapping.getEventClassifier());
				return (summary.getEventClasses().getClasses().size() < Short.MAX_VALUE);
			}
		}
		return false;
	}

	/**
	 * Return true if all replay inputs are correct: parameter type is correct and
	 * non empty (no null); all transitions are mapped to cost; all event classes
	 * (including dummy event class, i.e. an event class that does not exist in log,
	 * any transitions that are NOT silent and not mapped to any event class in the
	 * log is mapped to it) are mapped to cost; all costs should be non negative;
	 * numStates is non negative
	 */
	public boolean isParameterReqCorrect(PetrinetGraph net, XLog log, TransEvClassMapping mapping,
			IPNReplayParameter parameter) {
		if (parameter instanceof CostBasedCompleteParam) {
			CostBasedCompleteParam param = (CostBasedCompleteParam) parameter;
			if ((param.getMapTrans2Cost() != null) && (param.getMapEvClass2Cost() != null)
					&& (param.getInitialMarking() != null) && (param.getFinalMarkings() != null)) {
				// check all transitions are indeed mapped to cost
				if ((param.getMapTrans2Cost().keySet().containsAll(net.getTransitions()))) {
					Set<XEventClass> evClassWithCost = param.getMapEvClass2Cost().keySet();
					// check all event classes are mapped to cost
					XEventClassifier classifier = mapping.getEventClassifier();
					XLogInfo summary = XLogInfoFactory.createLogInfo(log, classifier);
					XEventClasses eventClassesName = summary.getEventClasses();

					if (evClassWithCost.containsAll(eventClassesName.getClasses())) {

						// all cost should be non negative
						for (Integer costVal : param.getMapEvClass2Cost().values()) {
							if (costVal < 0) {
								return false;
							}
						}
						for (Integer costVal : param.getMapTrans2Cost().values()) {
							if (costVal < 0) {
								return false;
							}
						}
						return true;
					}
				}
			}
		}
		return false;
	}

	private void importParameters(CostBasedCompleteParam parameters) {
		// replay parameters
		mapTrans2Cost = parameters.getMapTrans2Cost();
		//		maxNumOfStates = parameters.getMaxNumOfStates();
		mapEvClass2Cost = parameters.getMapEvClass2Cost();
		mapSync2Cost = parameters.getMapSync2Cost();
		mapTrans2Cost = parameters.getMapTrans2Cost();

		initMarking = parameters.getInitialMarking();
		finalMarkings = parameters.getFinalMarkings();

		numThreads = parameters.getNumThreads();

	}

}
