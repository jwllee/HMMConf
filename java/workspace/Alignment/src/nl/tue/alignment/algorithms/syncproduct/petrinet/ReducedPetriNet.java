package nl.tue.alignment.algorithms.syncproduct.petrinet;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.collections15.map.HashedMap;
import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetEdge;
import org.processmining.models.graphbased.directed.petrinet.PetrinetNode;
import org.processmining.models.graphbased.directed.petrinet.elements.Arc;
import org.processmining.models.graphbased.directed.petrinet.elements.InhibitorArc;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.ResetArc;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;

import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.procedure.TObjectIntProcedure;
import nl.tue.alignment.algorithms.syncproduct.GenericMap2Int;
import nl.tue.alignment.algorithms.syncproduct.petrinet.ReducedTransition.Type;

public class ReducedPetriNet {

	private final List<ReducedTransition> transitions;
	private List<ReducedPlace> places;

	private final TObjectIntMap<ReducedPlace> initialMarking;
	private final TObjectIntMap<ReducedPlace> finalMarking;

	public ReducedPetriNet(Petrinet net, XEventClasses classes, TObjectIntMap<Transition> trans2id,
			TObjectIntMap<XEventClass> c2id, TransEvClassMapping map, GenericMap2Int<Transition> mapTrans2Cost,
			GenericMap2Int<XEventClass> mapEvClass2Cost, GenericMap2Int<Transition> mapSync2Cost,
			Marking initialMarking, Marking finalMarking) {

		Map<Place, ReducedPlace> places = new HashedMap<>(net.getPlaces().size());

		this.places = new ArrayList<>(net.getPlaces().size());
		this.initialMarking = new TObjectIntHashMap<>(3, 0.5f, 0);
		this.finalMarking = new TObjectIntHashMap<>(3, 0.5f, 0);

		// copy the places and keep local map
		int i = 0;
		for (Place p : net.getPlaces()) {
			ReducedPlace rp = new ReducedPlace(i++);
			this.places.add(rp);
			places.put(p, rp);
			this.getInitialMarking().put(rp, initialMarking.occurrences(p));
			this.getFinalMarking().put(rp, finalMarking.occurrences(p));
		}

		this.transitions = new ArrayList<>(net.getTransitions().size());
		// then copy the transitions;
		for (Transition t : net.getTransitions()) {
			ReducedTransition rt;
			if (t.isInvisible() || c2id.get(map.get(t)) < 0) {
				rt = new ReducedTransition(trans2id.get(t), mapTrans2Cost.get(t));
			} else {
				rt = new ReducedTransition(trans2id.get(t), c2id.get(map.get(t)), mapTrans2Cost.get(t),
						mapSync2Cost.get(t));
			}
			transitions.add(rt);

			// for each transition copy the input
			for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> edge : net.getInEdges(t)) {
				Place p = (Place) edge.getSource();
				ReducedPlace rp = places.get(p);
				if (edge instanceof Arc) {
					int weight = ((Arc) edge).getWeight();
					rt.addToInput(rp, weight);
					rp.addToOutput(rt, weight);
				} else if (edge instanceof InhibitorArc) {
					rt.addToInput(rp, ReducedTransition.INHIBITOR);
					rp.addToOutput(rt, ReducedTransition.INHIBITOR);
				} else {
					throw new UnsupportedOperationException("Unknown Arc type in Petrinet input of transition " + t);
				}
			}

			// and the output
			for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> edge : net.getOutEdges(t)) {
				Place p = (Place) edge.getTarget();
				ReducedPlace rp = places.get(p);
				if (edge instanceof Arc) {
					int weight = ((Arc) edge).getWeight();
					rt.addToOutput(rp, weight);
					rp.addToInput(rt, weight);
				} else if (edge instanceof ResetArc) {
					throw new UnsupportedOperationException("Reset arcs are not yet supported.");
				} else {
					throw new UnsupportedOperationException("Unknown Arc type in Petrinet output of transition " + t);
				}
			}

		}

	}

	public boolean reduce(int maxStep, int maxLength) {
		int stepCount = 0;
		// try to find pairs of transitions which can be reduced.
		mainLoop: for (int i = 0; i < transitions.size(); i++) {
			boolean merged = false;
			for (int j = 0; j < transitions.size(); j++) {
				if (i == j) {
					continue;
				}
				// check if we can merge i and j.
				final ReducedTransition t1 = transitions.get(i);
				final ReducedTransition t2 = transitions.get(j);
				Type type = ReducedTransition.canMerge(maxLength, t1, t2);
				final ReducedTransition mergedTransition;
				switch (type) {
					case CHOICE :
						// compute the merged transition
						mergedTransition = ReducedTransition.mergeChoice(t1, t2);

						// correct places
						mergedTransition.forEachInputArc(new TObjectIntProcedure<ReducedPlace>() {
							public boolean execute(ReducedPlace a, int b) {
								assert a.getOutput().get(t1) == b;
								a.getOutput().remove(t1);
								assert a.getOutput().get(t2) == b;
								a.getOutput().remove(t2);
								a.getOutput().put(mergedTransition, b);
								return true;
							}
						});
						mergedTransition.forEachOutputArc(new TObjectIntProcedure<ReducedPlace>() {
							public boolean execute(ReducedPlace a, int b) {
								assert a.getInput().get(t1) == b;
								a.getInput().remove(t1);
								assert a.getInput().get(t2) == b;
								a.getInput().remove(t2);
								a.getInput().put(mergedTransition, b);
								return true;
							}
						});

						// insert transition into the list at position i
						transitions.set(i, mergedTransition);
						// remove element at index j
						transitions.remove(j);
						// decrement j;
						j--;
						i -= (i > j ? 1 : 0);
						// signal merge
						stepCount++;
						merged = true;
						break;
					case SEQUENCE :
						// compute the merged transition
						mergedTransition = ReducedTransition.mergeSequence(t1, t2);

						// correct places
						mergedTransition.forEachInputArc(new TObjectIntProcedure<ReducedPlace>() {
							public boolean execute(ReducedPlace a, int b) {
								assert a.getOutput().get(t1) == b;
								a.getOutput().remove(t1);
								a.getOutput().put(mergedTransition, b);
								return true;
							}
						});
						t1.forEachOutputArc(new TObjectIntProcedure<ReducedPlace>() {
							public boolean execute(ReducedPlace a, int b) {
								assert a.getInput().get(t1) == b;
								a.getInput().remove(t1);
								assert a.getOutput().get(t2) == b;
								a.getOutput().remove(t2);
								return true;
							}
						});
						mergedTransition.forEachOutputArc(new TObjectIntProcedure<ReducedPlace>() {
							public boolean execute(ReducedPlace a, int b) {
								assert a.getInput().get(t2) == b;
								a.getInput().remove(t2);
								a.getInput().put(mergedTransition, b);
								return true;
							}
						});

						// insert transition into the list at position i
						transitions.set(i, mergedTransition);
						// remove element at index j
						transitions.remove(j);
						// decrement j;
						j--;
						i -= (i > j ? 1 : 0);
						// signal merge
						stepCount++;
						merged = true;
						break;
					default :
						break;
				}
				if (stepCount >= maxStep) {
					break mainLoop;
				}
			}
			if (merged) {
				// something was merged, so let's start again.
				i = -1;
			}
		}
		// sort out the places.
		Set<ReducedPlace> placeSet = new HashSet<>();
		for (ReducedTransition t : transitions) {
			placeSet.addAll(t.getInput().keySet());
			placeSet.addAll(t.getOutput().keySet());
		}
		// new places
		this.places = new ArrayList<>(placeSet);
		return stepCount > 0;
	}

	public void toDot(final PrintStream out) {
		out.println("Digraph P {");
		for (ReducedPlace p : places) {
			out.print(p.toIdString());
			out.println(" [shape=\"circle\",label=<" + p.toHTMLString() + ">];");
		}
		for (final ReducedTransition t : transitions) {
			out.print(t.toIdString());
			out.println(" [shape=\"rectangle\",label=<" + t.toHTMLString() + ">];");

			t.forEachInputArc(new TObjectIntProcedure<ReducedPlace>() {

				public boolean execute(ReducedPlace a, int b) {
					out.print(a.toIdString());
					out.print(" -> ");
					out.print(t.toIdString());
					if (b != ReducedTransition.INHIBITOR) {
						out.println(" [label=\"" + (b > 1 ? b : "") + "\"];");
					} else {
						out.println(" [arrowtail=\"circle\",arrowhead=\"none\"];");
					}
					return true;
				}
			});

			t.forEachOutputArc(new TObjectIntProcedure<ReducedPlace>() {

				public boolean execute(ReducedPlace a, int b) {
					out.print(t.toIdString());
					out.print(" -> ");
					out.print(a.toIdString());
					out.println(" [label=\"" + (b > 1 ? b : "") + "\"];");
					return true;
				}
			});

		}
		out.println("}");
	}

	public List<ReducedTransition> getTransitions() {
		return transitions;
	}

	public List<ReducedPlace> getPlaces() {
		return places;
	}

	public TObjectIntMap<ReducedPlace> getInitialMarking() {
		return initialMarking;
	}

	public TObjectIntMap<ReducedPlace> getFinalMarking() {
		return finalMarking;
	}

}
