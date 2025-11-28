import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def add_recommendation_methods(analyzer):
    """Добавление методов для рекомендаций с учетом близости направлений"""
    
    def create_field_proximity_matrix(self):
        """Создание матрицы близости научных направлений"""
        # Определяем близость направлений (0-1, где 1 - максимальная близость)
        field_proximity = {
            0: {0: 1.0, 1: 0.3, 2: 0.2, 3: 0.1, 4: 0.4, 5: 0.7, 6: 0.8, 9: 0.1},  # AI
            1: {0: 0.3, 1: 1.0, 2: 0.1, 3: 0.7, 4: 0.6, 5: 0.4, 6: 0.2, 9: 0.1},  # Physics
            2: {0: 0.2, 1: 0.1, 2: 1.0, 3: 0.3, 4: 0.1, 5: 0.1, 6: 0.5, 9: 0.6},  # Biology
            3: {0: 0.1, 1: 0.7, 2: 0.3, 3: 1.0, 4: 0.2, 5: 0.3, 6: 0.1, 9: 0.4},  # Materials
            4: {0: 0.4, 1: 0.6, 2: 0.1, 3: 0.2, 4: 1.0, 5: 0.3, 6: 0.5, 9: 0.1},  # Mathematics
            5: {0: 0.7, 1: 0.4, 2: 0.1, 3: 0.3, 4: 0.3, 5: 1.0, 6: 0.4, 9: 0.1},  # Robotics
            6: {0: 0.8, 1: 0.2, 2: 0.5, 3: 0.1, 4: 0.5, 5: 0.4, 6: 1.0, 9: 0.2},  # Data Science
            9: {0: 0.1, 1: 0.1, 2: 0.6, 3: 0.4, 4: 0.1, 5: 0.1, 6: 0.2, 9: 1.0}   # Other
        }
        return field_proximity
    
    def create_researcher_vectors(self, results):
        """Создание векторов исследователей для рекомендательной системы"""
        print("🧮 Создание векторов исследователей...")
        
        vectors = []
        authors = []
        research_fields = []
        
        for result in results:
            # Создаем вектор: [цитируемость, публикации]
            citation_vec = result['predicted_citations']
            publication_vec = result['predicted_publications']
            field = result['predicted_field']
            
            main_vector = [citation_vec, publication_vec]
            
            vectors.append(main_vector)
            research_fields.append(field)
            authors.append(result['author'])
        
        return vectors, research_fields, authors
    
    def calculate_field_aware_similarity(self, vectors, research_fields, authors):
        """Расчет схожести с учетом близости научных направлений"""
        print("📊 Расчет схожести с учетом направлений...")
        
        # Нормализация числовых векторов
        scaler = StandardScaler()
        normalized_vectors = scaler.fit_transform(vectors)
        
        # Матрица близости направлений
        field_proximity = self.create_field_proximity_matrix()
        
        # Расчет попарной схожести с учетом направлений
        n_researchers = len(authors)
        similarity_matrix = np.zeros((n_researchers, n_researchers))
        
        for i in range(n_researchers):
            for j in range(n_researchers):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                    continue
                
                # Косинусное сходство по числовым признакам
                vec_i = normalized_vectors[i]
                vec_j = normalized_vectors[j]
                numerical_similarity = cosine_similarity([vec_i], [vec_j])[0][0]
                
                # Близость направлений
                field_i = research_fields[i]
                field_j = research_fields[j]
                field_similarity = field_proximity[field_i][field_j]
                
                # Комбинированная схожесть (взвешенная сумма)
                # Даем больше веса близости направлений (0.6) чем числовой схожести (0.4)
                combined_similarity = 0.4 * numerical_similarity + 0.6 * field_similarity
                
                similarity_matrix[i][j] = combined_similarity
        
        # Получение рекомендаций для каждого исследователя
        recommendations = {}
        
        for i, author in enumerate(authors):
            # Получаем индексы наиболее похожих исследователей (исключая самого себя)
            similar_indices = np.argsort(similarity_matrix[i])[::-1][1:11]  # Топ-10 для фильтрации
            
            # Фильтруем только близкие направления (порог близости > 0.3)
            author_recommendations = []
            for idx in similar_indices:
                if idx != i:
                    field_i = research_fields[i]
                    field_j = research_fields[idx]
                    field_sim = field_proximity[field_i][field_j]
                    
                    # Включаем только если направления достаточно близки
                    if field_sim >= 0.3:
                        similarity_score = similarity_matrix[i][idx]
                        
                        author_recommendations.append({
                            'author': authors[idx],
                            'similarity_score': float(similarity_score),
                            'field_similarity': float(field_sim),
                            'citation_impact': vectors[idx][0],
                            'publication_count': vectors[idx][1],
                            'research_field': research_fields[idx]
                        })
            
            # Берем топ-5 из отфильтрованных
            author_recommendations = sorted(author_recommendations, 
                                          key=lambda x: x['similarity_score'], 
                                          reverse=True)[:5]
            
            recommendations[author] = {
                'citation_impact': vectors[i][0],
                'publication_count': vectors[i][1],
                'research_field': research_fields[i],
                'nearest_neighbors': author_recommendations
            }
        
        return recommendations
    
    def generate_recommendations(self, results):
        """Генерация рекомендаций для исследователей с учетом направлений"""
        print("🎯 Генерация рекомендаций с учетом научных направлений...")
        
        # Создаем векторы
        vectors, research_fields, authors = self.create_researcher_vectors(results)
        
        # Рассчитываем схожесть с учетом направлений
        recommendations = self.calculate_field_aware_similarity(vectors, research_fields, authors)
        
        return recommendations
    
    def save_recommendations_to_json(self, recommendations, filename='sirius_recommendations.json'):
        """Сохранение рекомендаций в JSON файл"""
        print(f"💾 Сохранение рекомендаций в {filename}...")
        
        # Подготовка данных для JSON
        output_data = {}
        
        field_names = {
            0: "Искусственный интеллект",
            1: "Физика и квантовые технологии", 
            2: "Биология и генетика",
            3: "Химия и материалы",
            4: "Математика",
            5: "Робототехника",
            6: "Data Science",
            9: "Другое"
        }
        
        for author, data in recommendations.items():
            output_data[author] = {
                'citation_impact': float(data['citation_impact']),
                'publication_count': float(data['publication_count']),
                'research_field': int(data['research_field']),
                'research_field_name': field_names.get(data['research_field'], 'Другое'),
                'nearest_neighbors': []
            }
            
            for neighbor in data['nearest_neighbors']:
                output_data[author]['nearest_neighbors'].append({
                    'author': neighbor['author'],
                    'similarity_score': neighbor['similarity_score'],
                    'field_similarity': neighbor['field_similarity'],
                    'citation_impact': float(neighbor['citation_impact']),
                    'publication_count': float(neighbor['publication_count']),
                    'research_field': int(neighbor['research_field']),
                    'research_field_name': field_names.get(neighbor['research_field'], 'Другое')
                })
        
        # Сохранение в JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Рекомендации сохранены в {filename}")
        return output_data
    
    def analyze_and_recommend(self, results):
        """Полный анализ и генерация рекомендаций с учетом направлений"""
        print("\n🔍 ЗАПУСК СИСТЕМЫ РЕКОМЕНДАЦИЙ С УЧЕТОМ НАПРАВЛЕНИЙ")
        print("=" * 60)
        
        # Генерация рекомендаций
        recommendations = self.generate_recommendations(results)
        
        # Сохранение в JSON
        json_data = self.save_recommendations_to_json(recommendations)
        
        # Вывод результатов
        print("\n🎯 РЕЗУЛЬТАТЫ РЕКОМЕНДАЦИЙ")
        print("=" * 60)
        
        field_names = {
            0: "🖥️ Искусственный интеллект",
            1: "⚛️ Физика и квантовые технологии", 
            2: "🧬 Биология и генетика",
            3: "🔬 Химия и материалы",
            4: "📐 Математика",
            5: "🤖 Робототехника",
            6: "📊 Data Science",
            9: "🔍 Другое"
        }
        
        for author, data in json_data.items():
            print(f"\n👨‍🔬 Исследователь: {author}")
            print(f"📈 Цитируемость: {data['citation_impact']:.0f}")
            print(f"🎯 Направление: {field_names.get(data['research_field'], 'Другое')}")
            print(f"📊 Публикации: {data['publication_count']:.0f}")
            print("👥 Ближайшие коллеги (похожие направления):")
            
            for i, neighbor in enumerate(data['nearest_neighbors'], 1):
                print(f"   {i}. {neighbor['author']}")
                print(f"      Схожесть: {neighbor['similarity_score']:.3f}")
                print(f"      Схожесть направлений: {neighbor['field_similarity']:.3f}")
                print(f"      Направление: {neighbor['research_field_name']}")
            print("-" * 50)
        
        return json_data
    
    # Добавляем методы к классу
    analyzer.create_field_proximity_matrix = create_field_proximity_matrix.__get__(analyzer)
    analyzer.create_researcher_vectors = create_researcher_vectors.__get__(analyzer)
    analyzer.calculate_field_aware_similarity = calculate_field_aware_similarity.__get__(analyzer)
    analyzer.generate_recommendations = generate_recommendations.__get__(analyzer)
    analyzer.save_recommendations_to_json = save_recommendations_to_json.__get__(analyzer)
    analyzer.analyze_and_recommend = analyze_and_recommend.__get__(analyzer)
    
    return analyzer

print("✅ Методы для рекомендаций с учетом направлений добавлены!")