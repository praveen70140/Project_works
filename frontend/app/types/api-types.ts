export type feedbackApi = { feedback_prediction: string };
export type salesApi = { high_sales_prediction: number };
export type monthlySalesApi = { rf_sales_prediction: number };
export type ratingApi = { rf_rating_prediction: number };
export type successProbabilityApi = {
  success_probability_percentage: number;
  is_successful: boolean;
};
export type cityRecommendationApi = {
  top_3_recommendations: {
    city: string;
    probability_percent: number;
  }[];
};
export type monthRecommendationApi = {
  top_3_month_recommendations: {
    month: number;
    probability_percent: number;
  }[];
};
export type healthApi = {
  status: string;
  models_loaded: string[];
};
type marketMatrix = {
  market_matrix: {
    [k: string]: {
      [l: string]: number;
    };
  };
  city_global_probabilities: {
    [k: string]: number;
  };
};

export type unifiedApi = {
  feedback_prediction: feedbackApi;
  high_sales_prediction: salesApi;
  rf_rating_prediction: ratingApi;
  rf_monthly_sales: monthlySalesApi;
  rf_success_prob: successProbabilityApi;
  market_matrix: marketMatrix;
  gemini_recommendation: string;
};
