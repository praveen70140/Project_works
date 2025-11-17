"use client";

import { Card, CardBody, Tab, Tabs } from "@heroui/react";
import React, { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { mainFormData } from "../types/main-form-data";
import MainForm from "../components/main-form";
import Result from "../components/result";
import { parseDate } from "@internationalized/date";

const HomePage = () => {
  const methods = useForm<mainFormData>({
    defaultValues: {
      city: "Bengaluru",
      cuisine: "Indian",
      location: "Indiranagar",
      date: parseDate("2025-11-17"),
      rating: 5.0,
      restaurantName: "Best Restaurant",
      salesAmount: 400,
      salesQuantity: 4000,
    },
  });
  const [tabState, setTabState] = useState<string>("form");

  const {
    formState: { isSubmitted },
  } = methods;

  const onSubmit = () => {
    setTabState("result");
  };

  return (
    <div className="p-3 flex flex-col space-y-4">
      <h2 className="text-primary-600 text-2xl font-semibold">Welcome!</h2>
      <FormProvider {...methods}>
        <Tabs
          aria-label="Steps"
          selectedKey={tabState}
          onSelectionChange={(e) => setTabState(String(e))}
        >
          <Tab key="form" title="Form">
            <Card>
              <CardBody>
                <MainForm onSubmit={onSubmit} />
              </CardBody>
            </Card>
          </Tab>
          <Tab key="result" title="Result" isDisabled={!isSubmitted}>
            <Card>
              <CardBody>
                <Result />
              </CardBody>
            </Card>
          </Tab>
        </Tabs>
      </FormProvider>
    </div>
  );
};

export default HomePage;
