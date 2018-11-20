package gradient.updater;

import java.util.Arrays;

public class SimpleGradientUpdater implements IGradientBasedUpdater{

	private double[] learningRates;

	public SimpleGradientUpdater(double learningRate, int variableCount) {
		this(new double[variableCount]);
		Arrays.fill(learningRates, learningRate);
	}

	public SimpleGradientUpdater(double[] learningRates) {
		this.learningRates = learningRates;
	}
	
	@Override
	public void update(double[] values, double[] gradients, double fintess, double value) {
		for(int i=0; i<values.length;i++){
			values[i]-=gradients[i]*learningRates[i];
		}
	}

}
