# MediClaims AI - Enhanced Architecture Diagram Generation Prompt

## INSTRUCTION FOR DIAGRAM GENERATION

Create a professional technical architecture diagram in the EXACT SAME STYLE as the reference image provided, but showing the new Pre-Submission and Post-Submission flows. The diagram should maintain the same visual design language, component styling, and layout approach.

---

## VISUAL STYLE REQUIREMENTS (Match Reference Exactly)

**Layout Style:**
- Horizontal flow from left to right
- Clean, modern flat design with rounded rectangles
- Grouped sections with colored borders and background fills
- Clear directional arrows with descriptive labels
- Component icons/symbols within boxes where appropriate

**Color Scheme (Match Reference):**
- Main orchestrator section: Light yellow/cream background (#FFF3CD) with orange border
- Agent components: Light blue/teal background (#E0F7FF) 
- External systems: Orange background (#FFE4B5) with darker borders
- Data sources: Blue cylinders for databases
- Arrows: Dark gray/black with clear labels

**Typography:**
- Clean, sans-serif font (similar to reference)
- Component names in bold
- Connection labels in smaller, regular weight
- Hierarchical text sizing

---

## ARCHITECTURE COMPONENTS TO INCLUDE

### LEFT SIDE - USER INTERFACE
```
Web Dashboard → API Server
(Same style as reference, green monitor + blue server boxes)
```

### MAIN ORCHESTRATOR SECTION (Large yellow box like reference)
**Title:** "ENHANCED AGENTIC ORCHESTRATOR"

**Internal Components (arranged vertically like reference AI AGENTS section):**

**PRE-SUBMISSION LAYER** (Light blue background)
- Claim Intake Adapter
- Pre-Submission Analyzer  
- Risk Predictor (with shield icon like reference)
- Denial Reason Predictor
- Compliance & Rules Engine
- Recommendation Engine
- Auto Corrector (with wrench/tool icon)

**PROCESSING CORE** (Center section)
- LangGraph Workflow Engine
- Claim Submitter (with send icon like reference)

**POST-SUBMISSION LAYER** (Light orange background)  
- ERA/835 Ingestion Service
- Denial Code Mapper
- Denial Taxonomy Engine
- Appeal Explanation Engine
- Appeal Generator (with document icon)
- Resubmission Coordinator
- Feedback Learner (with brain/learning icon)

### RIGHT SIDE SECTIONS

**MCP SERVER** (Same position and style as reference)
- Connected to main orchestrator
- Arrow labeled "fetch/process data"
- Connected to "Hospital CRM" database (blue cylinder)

**PAYER APIS** (Orange section like reference INSURANCE APIs)
- Aetna (with building icon)
- United (with building icon) 
- BlueCross (with building icon)
- Additional: "ERA/835 Feed" (new)

### DATA SOURCES (Bottom section, blue cylinders)
- OpenEMR Database
- CSV Fallback Store  
- ERA Repository
- Denial Taxonomy Store
- Feature Store
- Denial Patterns KB

---

## CONNECTION FLOWS (Labeled Arrows)

### PRE-SUBMISSION FLOW (Blue arrows)
1. `Web Dashboard` → `API Server` → `Claim Intake Adapter` 
2. `Claim Intake Adapter` → `Pre-Submission Analyzer` (label: "raw claim data")
3. `Pre-Submission Analyzer` → `Risk Predictor` (label: "risk assessment")
4. `Pre-Submission Analyzer` → `Denial Reason Predictor` (label: "predict denials")
5. `Risk Predictor` + `Denial Reason Predictor` → `Compliance & Rules Engine` (label: "validation check")
6. `Compliance & Rules Engine` → `Recommendation Engine` (label: "compliance result")
7. `Recommendation Engine` → `Auto Corrector` (label: "auto-fix issues")
8. `Auto Corrector` → `Web Dashboard` (label: "human review needed", dashed line)
9. `Recommendation Engine` → `Claim Submitter` (label: "submit approved")
10. `Claim Submitter` → `Payer APIs` (label: "submit claims")

### POST-SUBMISSION FLOW (Orange arrows)  
11. `Payer APIs` → `ERA/835 Ingestion Service` (label: "ERA/835 response")
12. `ERA/835 Ingestion Service` → `Denial Code Mapper` (label: "parse denials")
13. `Denial Code Mapper` → `Denial Taxonomy Engine` (label: "categorize reasons")
14. `Denial Taxonomy Engine` → `Appeal Explanation Engine` (label: "explain denials")
15. `Appeal Explanation Engine` → `Appeal Generator` (label: "draft appeal")
16. `Appeal Generator` → `Web Dashboard` (label: "human review/edit", dashed line)
17. `Web Dashboard` → `Resubmission Coordinator` (label: "approve appeal")
18. `Resubmission Coordinator` → `Payer APIs` (label: "resubmit appeal")

### LEARNING FEEDBACK LOOP (Purple arrows)
19. `Payer APIs` → `Feedback Learner` (label: "final outcomes")
20. `Feedback Learner` → `Feature Store` + `Denial Patterns KB` (label: "update patterns")
21. `Feature Store` + `Denial Patterns KB` → `Pre-Submission Analyzer` + `Denial Reason Predictor` (label: "improve predictions", curved arrow)

### DATA ACCESS (Gray dotted lines)
- MCP Server connections to all data sources
- Components reading from data sources as needed

---

## TECHNICAL ANNOTATIONS

Add small technical labels under each major component:
- `Pre-Submission Analyzer`: "Python 3.13 + LangGraph"
- `Risk Predictor`: "Azure OpenAI GPT-4"  
- `Denial Reason Predictor`: "ML Models + Rules Engine"
- `Compliance & Rules Engine`: "HIPAA + State Regulations"
- `Appeal Generator`: "GPT-4 + Template Engine"
- `ERA/835 Ingestion`: "EDI Parser + Python"
- `MCP Server`: "FastAPI + Tool Integration"
- `Payer APIs`: "REST APIs + EDI 837/835"

---

## GROUPING AND SECTIONS

**Main Groups (with colored borders like reference):**
1. **User Interface** (Green section, left)
2. **Enhanced Agentic Orchestrator** (Yellow section, center - largest)  
   - **Pre-Submission** (light blue subsection)
   - **Core Processing** (neutral subsection)  
   - **Post-Submission** (light orange subsection)
3. **MCP Server** (Blue section, right-center)
4. **Payer APIs** (Orange section, right)  
5. **Data Sources** (Gray section, bottom)

---

## FLOW PHASES VISUALIZATION

Add phase labels/headers:
- **"PRE-SUBMISSION PHASE"** (above pre-submission components)
- **"POST-SUBMISSION PHASE"** (above post-submission components)  
- **"CONTINUOUS LEARNING"** (near feedback loop)

---

## FINAL DESIGN NOTES

- Maintain the same clean, professional aesthetic as the reference
- Use consistent spacing and alignment
- Ensure all text is readable and well-sized
- Use the same icon style (simple, flat, meaningful)
- Keep the same arrow style and thickness
- Maintain visual hierarchy with proper grouping
- Add subtle shadows/depth like the reference
- Ensure the diagram fits well in a landscape orientation

The result should look like a natural evolution of your current diagram, maintaining the same visual DNA while clearly showing the enhanced Pre/Post-submission architecture.
