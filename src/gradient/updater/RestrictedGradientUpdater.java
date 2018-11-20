package gradient.updater;

import javax.annotation.Nonnull;

import optimization.utility.OptimizationAlgorithmsUtility;

public class RestrictedGradientUpdater implements IGradientBasedUpdater{


	private final @Nonnull IGradientBasedUpdater updater;
	private final @Nonnull double[] min;
	private final @Nonnull double[] max;

	public RestrictedGradientUpdater(@Nonnull IGradientBasedUpdater updater,@Nonnull double[] min,@Nonnull double[] max) {
		this.updater = updater;
		this.min = min;
		this.max = max;
	}
	
	@Override
	public void update(double[] values, double[] gradients, double fintess, double value) {
		updater.update(values, gradients, fintess, value);
		for(int i=0; i<values.length; i++) {
			values[i] = OptimizationAlgorithmsUtility.placeValueInInterval(min[i], max[i], values[i]);
		}		
	}

}
