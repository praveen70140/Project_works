"use client";

import {
  addToast,
  Button,
  Card,
  CardBody,
  CardFooter,
  CardHeader,
  Input,
} from "@heroui/react";
import React, { useState } from "react";
import loginData from "@/app/data/login.json";
import { SubmitHandler, useForm } from "react-hook-form";
import { useRouter } from "next/navigation";
import { EyeClosedIcon, EyeIcon } from "lucide-react";

type Inputs = {
  email: string;
  password: string;
};

const Login = () => {
  const [isVisible, setIsVisible] = useState(false);
  const { push } = useRouter();

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<Inputs>();

  const onSubmit: SubmitHandler<Inputs> = (data) => {
    const allowedCredentials: Inputs[] = loginData;

    if (
      allowedCredentials.some(
        (example) =>
          example.email === data.email && example.password === data.password
      )
    ) {
      push("/home");
    } else {
      const example = allowedCredentials[0];
      addToast({
        title: "Incorrect credentials!",
        color: "danger",
        timeout: 5000,
        endContent: (
          <Button
            variant="ghost"
            color="danger"
            onPress={() => {
              setValue("email", example.email);
              setValue("password", example.password);
            }}
          >
            Autofill
          </Button>
        ),
        description: `Try email "${example.email}" and password "${example.password}"`,
      });
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit(onSubmit)}>
        <Card className="w-96 p-5" fullWidth>
          <CardHeader>
            <h2 className=" self-center text-3xl font-medium text-primary-500">
              Welcome!
            </h2>
          </CardHeader>
          <CardBody className="space-y-3">
            <p className="text-default-400 text-center">Please log in</p>
            <Input
              variant="bordered"
              size="md"
              label="Email"
              type="email"
              {...register("email", { required: true })}
            />
            <Input
              variant="bordered"
              size="md"
              label="Password"
              type={isVisible ? "text" : "password"}
              {...register("password", { required: true })}
              endContent={
                <Button
                  isIconOnly
                  variant="light"
                  onPress={() => setIsVisible((e) => !e)}
                >
                  {isVisible ? <EyeIcon /> : <EyeClosedIcon />}
                </Button>
              }
            />
          </CardBody>
          <CardFooter>
            <Button fullWidth color="primary" type="submit">
              Submit
            </Button>
          </CardFooter>
        </Card>
      </form>
    </div>
  );
};

export default Login;
