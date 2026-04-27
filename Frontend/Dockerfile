FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 5000

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY ["FaceRecognition_Frontend/FaceRecognition_Frontend.csproj", "FaceRecognition_Frontend/"]
RUN dotnet restore "FaceRecognition_Frontend/FaceRecognition_Frontend.csproj"

COPY . .
WORKDIR "/src/FaceRecognition_Frontend"
RUN dotnet build "FaceRecognition_Frontend.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "FaceRecognition_Frontend.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "FaceRecognition_Frontend.dll"]
