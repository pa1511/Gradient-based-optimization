package gradient.updater;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public class PerDimensionGradientUpdater implements IGradientBasedUpdater{

	private final @Nonnull double[] learningRates;	
	private final @Nonnegative int dimensionRepeatFactor;
	private @Nonnegative int dimension = 0;
	private @Nonnegative int repeat = 0;

	public PerDimensionGradientUpdater(@Nonnegative double[] learningRate, @Nonnegative int dimensionRepeatFactor) {
		this.learningRates = learningRate;
		this.dimensionRepeatFactor = dimensionRepeatFactor;
	}
	
	@Override
	public void update(@Nonnull double[] values,@Nonnull double[] gradients, double fintess, double value) {
		repeat++;
		if(repeat%dimensionRepeatFactor==0)
			dimension = (dimension+1)%gradients.length;
		values[dimension]-=gradients[dimension]*learningRates[dimension];
	}

}
