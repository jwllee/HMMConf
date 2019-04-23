package nl.tue.alignment.test;

import java.util.HashSet;
import java.util.Set;

import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetEdge;
import org.processmining.models.graphbased.directed.petrinet.PetrinetNode;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;

public class BPDC2019 {

	public static void main(String[] args) {
		Petrinet net = BasicCodeSnippet.constructNet("c://temp/net4a-22February.pnml");

		for (Transition t : (Transition[]) net.getTransitions().toArray(new Transition[0])) {
			if (!t.getLabel().startsWith("q") && !t.getLabel().startsWith("y") && !t.getLabel().startsWith("aq")
					&& !t.getLabel().startsWith("b")) {
				for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> a : net.getInEdges(t)) {
					net.removeArc(a.getSource(), t);
				}
				for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> a : net.getOutEdges(t)) {
					net.removeArc(t, a.getTarget());
				}

				net.removeTransition(t);

			}
		}
		for (Place p : (Place[]) net.getPlaces().toArray(new Place[0])) {
			if (net.getInEdges(p).isEmpty() || net.getOutEdges(p).isEmpty()) {
				net.removePlace(p);
			}
		}
		for (Place p : (Place[]) net.getPlaces().toArray(new Place[0])) {
			for (Place p2 : (Place[]) net.getPlaces().toArray(new Place[0])) {
				if (p != p2 && getInput(net, p).equals(getInput(net, p2))
						&& getOutput(net, p).equals(getOutput(net, p2))) {
					net.removePlace(p2);
				}
			}
		}

		System.out.println("Digraph d{");
		for (Transition t : net.getTransitions()) {
			System.out.println("\"" + t.getLabel() + "\" [shape=box];");
			for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> a : net.getOutEdges(t)) {
				System.out.println("\"" + t.getLabel() + "\" -> \"" + a.getTarget().getLabel() + "\";");
			}
			for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> a : net.getInEdges(t)) {
				System.out.println("\"" + a.getSource().getLabel() + "\" -> \"" + t.getLabel() + "\";");
			}
		}
		System.out.println("}");

	}

	private static Set<Transition> getOutput(Petrinet net, Place p) {
		Set<Transition> set = new HashSet<>();
		for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> a : net.getOutEdges(p)) {
			set.add((Transition) a.getTarget());
		}
		return set;
	}

	private static Set<Transition> getInput(Petrinet net, Place p) {
		Set<Transition> set = new HashSet<>();
		for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> a : net.getInEdges(p)) {
			set.add((Transition) a.getSource());
		}
		return set;
	}
}
